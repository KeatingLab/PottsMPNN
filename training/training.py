"""
Code in this file was originally written by or adapted from:

    1) Dauparas et al. "Robust deep learning-based protein sequence design using ProteinMPNN"
    Science, 2022. doi: 10.1126/science.add2187
        - Code: https://github.com/dauparas/ProteinMPNN

    2) Birnbaum and Keating "Beyond native sequence recovery: Improved modeling of the
    sequence-energy landscape of protein structures"
    bioRxiv, 2026. doi:10.64898/2026.01.14.699067
       - Code: https://github.com/KeatingLab/PottsMPNN
"""

import os

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import argparse
import multiprocessing as mp
import pickle
import sys
import time
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

# The model and the Potts losses live in the repository root, one level up.
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_utils import loss_nll, loss_smoothed
from potts_mpnn_utils import PottsMPNN, nlcpl, potts_singlesite_loss
from utils import (PDB_dataset, StructureDataset, StructureSampler,
                   build_training_clusters, get_pdbs, get_std_opt, loader_pdb,
                   worker_init_fn)

# Single sequences use the 20 amino acids plus X for unknown residues. MSA
# sequences additionally use '-' for gaps, which the model treats as a 22nd token.
SEQ_ALPHABET = 'ACDEFGHIKLMNPQRSTVWYX'
MSA_ALPHABET = 'ACDEFGHIKLMNPQRSTVWYX-'


def load_pickle_stream(path):
    """Yield the entries of a file written as a sequence of pickled chunks."""
    with open(path, "rb") as f:
        while True:
            try:
                chunk = pickle.load(f)
                if isinstance(chunk, list):
                    yield from chunk
                else:
                    yield chunk
            except EOFError:
                break


def submit_worker(executor, loader, i_worker, data_source, base_folder, args):
    """Queue one data-preparation worker, which writes a pickled shard to disk."""
    return executor.submit(
        get_pdbs, loader, i_worker, data_source, base_folder,
        repeat=1,
        max_length=args.max_protein_length,
        num_units=args.num_examples_per_epoch,
        consensus_seqs=args.consensus_seqs,
        msa_match_dict_path=args.msa_match_dict,
        complex_mapping_path=args.complex_mapping_path,
        msa_dir=args.msa_dir,
        msa_seqs=args.msa_seqs,
        single_species_sample=args.single_species_sample,
        remove_missing=args.remove_missing,
        id_thresh=args.id_thresh,
        del_thresh=args.del_thresh,
        insrt_thresh=args.insrt_thresh,
    )


def start_workers(executor, train_loader, valid_loader, base_folder, args, n_pairs=2):
    """Queue `n_pairs` train shards and `n_pairs` valid shards."""
    train_futures = {}
    valid_futures = {}
    for i in range(n_pairs):
        train_futures[submit_worker(executor, train_loader, i, 'train', base_folder, args)] = i
        valid_futures[submit_worker(executor, valid_loader, i, 'valid', base_folder, args)] = i
    return train_futures, valid_futures


def get_one_pair(executor, train_futures, valid_futures,
                 train_loader, valid_loader, base_folder, args, requeue=False):
    """Block until one train and one valid shard are ready, then consume them.

    Each worker writes its shard to `<base_folder>/<worker>_data_<source>.pkl`, so
    the file is removed once it has been read and the worker index is free to be
    reused by a replacement worker.
    """
    done_train, _ = wait(train_futures.keys(), return_when=FIRST_COMPLETED)
    done_valid, _ = wait(valid_futures.keys(), return_when=FIRST_COMPLETED)

    train_future = done_train.pop()
    valid_future = done_valid.pop()

    train_file = train_future.result()
    valid_file = valid_future.result()

    pdb_dict_train = list(load_pickle_stream(train_file))
    os.remove(train_file)
    pdb_dict_valid = list(load_pickle_stream(valid_file))
    os.remove(valid_file)

    train_i = train_futures.pop(train_future)
    valid_i = valid_futures.pop(valid_future)
    if requeue:
        train_futures[submit_worker(executor, train_loader, train_i, 'train', base_folder, args)] = train_i
        valid_futures[submit_worker(executor, valid_loader, valid_i, 'valid', base_folder, args)] = valid_i

    return pdb_dict_train, pdb_dict_valid


def filter_clusters(clusters, exclude):
    """Drop chains whose chain label or PDB ID appears in `exclude`."""
    if not exclude:
        return clusters
    exclude = set(entry.lower() for entry in exclude)
    filtered = {}
    for cluster, items in clusters.items():
        kept = [item for item in items
                if item[0].lower() not in exclude
                and item[0].split('_')[0].lower() not in exclude]
        if kept:
            filtered[cluster] = kept
    return filtered


def build_loader(pdb_dict, args, alphabet, device, loader_kwargs):
    """Wrap a list of parsed structures in a length-clustered batch loader."""
    dataset = StructureDataset(pdb_dict, truncate=None,
                               max_length=args.max_protein_length, alphabet=alphabet)
    sampler = StructureSampler(dataset, batch_size=args.batch_size, device=device,
                               msa_seqs=args.msa_seqs, msa_batch_size=args.msa_batch_size)
    loader = DataLoader(dataset, batch_sampler=sampler, collate_fn=sampler.package,
                        pin_memory=True, **loader_kwargs)
    return sampler, loader


def prepare_batch(batch, device):
    """Move a featurized batch onto `device`.

    When MSA sequences are used, `featurize` samples several sequences for the one
    structure in the batch and only expands S/S_true/mask along the batch
    dimension. The structure tensors are expanded here so that every tensor shares
    a batch dimension, which the gather operations in the model require.
    """
    (X, S, S_true, mask, _lengths, chain_M, residue_idx, _mask_self,
     chain_encoding_all, _all_chain_lens, names) = batch

    X = X.to(device=device)
    S = S.to(device=device)
    S_true = S_true.to(device=device)
    mask = mask.to(device=device)
    chain_M = chain_M.to(device=device)
    residue_idx = residue_idx.to(device=device)
    chain_encoding_all = chain_encoding_all.to(device=device)

    n_seqs = S.shape[0]
    if X.shape[0] != n_seqs:
        X = X.expand(n_seqs, -1, -1, -1)
        chain_M = chain_M.expand(n_seqs, -1)
        residue_idx = residue_idx.expand(n_seqs, -1)
        chain_encoding_all = chain_encoding_all.expand(n_seqs, -1)

    return X, S, S_true, mask, chain_M, residue_idx, chain_encoding_all, names


def compute_train_loss(args, log_probs, etab, E_idx, S_true, mask, mask_for_loss, vocab):
    """Assemble the loss that is back-propagated for one batch.

    Returns `(loss, nlcpl_loss)`. `loss` is None when the batch has no scorable
    Potts edges, in which case the batch is skipped.
    """
    nlcpl_loss = None
    if args.etab_loss or args.etab_loss_only:
        nlcpl_loss, _, n_potts_edges = nlcpl(etab, E_idx, S_true, mask,
                                             fixed_denom=args.fixed_potts_denom)
        if n_potts_edges == 0:
            return None, None

    if args.etab_loss_only:
        loss = nlcpl_loss
    elif args.etab_singlesite_loss:
        _, loss = potts_singlesite_loss(etab, E_idx, S_true, mask_for_loss, vocab,
                                        fixed_denom=args.fixed_denom)
    else:
        _, loss = loss_smoothed(S_true, log_probs, mask_for_loss, vocab=vocab,
                                fixed_denom=args.fixed_denom)
        if args.etab_loss:
            loss = loss + nlcpl_loss

    return loss, nlcpl_loss


def save_checkpoint(path, model, optimizer, args, epoch, total_step, vocab):
    """Write a checkpoint with enough metadata to rebuild the model for inference."""
    torch.save({
        'epoch': epoch,
        'step': total_step,
        'num_edges': args.num_neighbors,
        'noise_level': args.backbone_noise,
        'hidden_dim': args.hidden_dim,
        'potts_dim': args.output_dim,
        'num_layers': args.num_encoder_layers,
        'num_encoder_layers': args.num_encoder_layers,
        'num_decoder_layers': args.num_decoder_layers,
        'vocab': vocab,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.optimizer.state_dict(),
    }, path)


def format_float(value):
    return np.format_float_positional(np.float32(value), unique=False, precision=3)


def main(args):
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    scaler = torch.cuda.amp.GradScaler(enabled=args.mixed_precision)

    base_folder = time.strftime(args.path_for_outputs, time.localtime())
    if base_folder[-1] != '/':
        base_folder += '/'
    os.makedirs(base_folder + 'model_weights', exist_ok=True)

    PATH = args.previous_checkpoint

    logfile = base_folder + 'log.txt'
    if not PATH:
        with open(logfile, 'w') as f:
            f.write('Epoch\tTrain\tValidation\n')

    data_path = args.path_for_training_data
    data_meta_path = args.path_for_meta_data
    if (len(args.consensus_seqs) > 0) or ('ingr' in args.msa_dir or 'ing_msas' in args.msa_dir):
        consensus_tag = '_consensus'
    else:
        consensus_tag = ''
    params = {
        "LIST"    : f"{data_meta_path}/list{consensus_tag}.csv",
        "VAL"     : f"{data_meta_path}/valid_clusters.txt",
        "TEST"    : f"{data_meta_path}/test_clusters.txt",
        "DIR"     : f"{data_path}",
        "DATCUT"  : "2030-Jan-01",
        "RESCUT"  : args.rescut, #resolution cutoff for PDBs
        "HOMO"    : 0.70, #min seq.id. to detect homo chains
        "CATH"    : "ingraham" in data_path
    }

    LOAD_PARAM = {'batch_size': 1,
                  'shuffle': True,
                  'pin_memory': False,
                  'num_workers': 2}

    if args.debug:
        args.num_examples_per_epoch = 500
        args.max_protein_length = 1000
        args.batch_size = 1

    print('loaded args')
    print(args)

    # Chains to hold out of training, e.g. membrane proteins for a soluble model or
    # chains whose MSA could not be assembled.
    exclude = []
    if args.soluble_mpnn:
        exclude_df = pd.read_csv(args.soluble_mpnn)
        exclude += list(exclude_df['PDB_IDS'].values)
    if args.exclude_msa:
        with open(args.exclude_msa, 'r') as f:
            exclude += [line.strip() for line in f if line.strip()]

    train, valid, test = build_training_clusters(params, args.debug)
    train = filter_clusters(train, exclude)
    valid = filter_clusters(valid, exclude)
    test = filter_clusters(test, exclude)
    print(f'built clusters: {len(train)} train, {len(valid)} valid, {len(test)} test')

    if args.debug:
        train_list = list(train.keys())[:100]
        valid_list = list(valid.keys())[:100]
        print('debug len: ', len(train_list), len(valid_list))
    else:
        train_list = list(train.keys())
        valid_list = list(valid.keys())

    train_set = PDB_dataset(train_list, loader_pdb, train, params)
    train_loader = DataLoader(train_set, worker_init_fn=worker_init_fn, **LOAD_PARAM)
    valid_set = PDB_dataset(valid_list, loader_pdb, valid, params)
    valid_loader = DataLoader(valid_set, worker_init_fn=worker_init_fn, **LOAD_PARAM)

    print('loaded loaders')

    # MSA sequences add a gap token to the amino acid alphabet.
    vocab = 22 if args.msa_seqs else 21
    alphabet = MSA_ALPHABET if args.msa_seqs else SEQ_ALPHABET
    print('vocab: ', vocab)

    model = PottsMPNN(num_letters=vocab,
                      vocab=vocab,
                      node_features=args.hidden_dim,
                      edge_features=args.hidden_dim,
                      hidden_dim=args.hidden_dim,
                      potts_dim=args.output_dim,
                      num_encoder_layers=args.num_encoder_layers,
                      num_decoder_layers=args.num_decoder_layers,
                      k_neighbors=args.num_neighbors,
                      dropout=args.dropout,
                      augment_eps=args.backbone_noise)
    model.to(device)

    print('setup model')
    if PATH:
        checkpoint = torch.load(PATH, map_location=device, weights_only=False)
        total_step = checkpoint['step'] #write total_step from the checkpoint
        epoch = checkpoint['epoch'] #write epoch from the checkpoint
        model.load_state_dict(checkpoint['model_state_dict'], strict=args.strict)
    else:
        total_step = 0
        epoch = 0

    optimizer = get_std_opt(model.parameters(), args.hidden_dim, total_step)

    if PATH and args.load_optimizer:
        optimizer.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    print('setup optimizer')

    loader_kwargs = {'num_workers': args.num_workers}
    executor = ProcessPoolExecutor(max_workers=2 * args.num_data_workers,
                                   mp_context=mp.get_context('spawn'))
    train_futures, valid_futures = start_workers(executor, train_loader, valid_loader,
                                                 base_folder, args,
                                                 n_pairs=args.num_data_workers)
    print('waiting to get result')

    pdb_dict_train, pdb_dict_valid = get_one_pair(executor, train_futures, valid_futures,
                                                  train_loader, valid_loader, base_folder,
                                                  args, requeue=True)

    train_sampler, loader_train = build_loader(pdb_dict_train, args, alphabet, device, loader_kwargs)
    valid_sampler, loader_valid = build_loader(pdb_dict_valid, args, alphabet, device, loader_kwargs)

    reload_c = 0
    best_val_loss = np.inf
    for e in range(args.num_epochs):
        t0 = time.time()
        e = epoch + e

        if e % args.reload_data_every_n_epochs == 0:
            if reload_c != 0:
                pdb_dict_train, pdb_dict_valid = get_one_pair(executor, train_futures, valid_futures,
                                                              train_loader, valid_loader, base_folder,
                                                              args, requeue=True)
                train_sampler, loader_train = build_loader(pdb_dict_train, args, alphabet, device, loader_kwargs)
                valid_sampler, loader_valid = build_loader(pdb_dict_valid, args, alphabet, device, loader_kwargs)
            reload_c += 1

        # The epoch seeds the MSA sequence sampling, so it has to be set before the
        # loader spawns its workers and pickles the sampler.
        train_sampler._set_epoch(e)
        valid_sampler._set_epoch(e)

        model.train()
        train_sum, train_acc, train_weights = 0., 0., 0.
        train_nlcpl_sum, train_nlcpl_batches = 0., 0

        pbar = tqdm(loader_train, desc=f"Epoch {e+1} [train]", unit="batch", miniters=100)
        for i_train_batch, batch in enumerate(pbar):
            X, S, S_true, mask, chain_M, residue_idx, chain_encoding_all, names = prepare_batch(batch, device)
            if mask.sum(dim=1).min() == 0:
                print(f"{names} have no valid positions")
            mask_for_loss = mask * chain_M
            randn = torch.randn(chain_M.shape, device=device)

            optimizer.zero_grad()
            with torch.cuda.amp.autocast(enabled=args.mixed_precision):
                log_probs, etab, E_idx = model(X, S, mask, chain_M, residue_idx,
                                               chain_encoding_all, randn)
                loss_av_smoothed, nlcpl_loss = compute_train_loss(
                    args, log_probs, etab, E_idx, S_true, mask, mask_for_loss, vocab)

            if loss_av_smoothed is None:
                continue
            if not torch.isfinite(loss_av_smoothed):
                print(f"skipping {names}: loss is not finite")
                continue

            scaler.scale(loss_av_smoothed).backward()
            if args.gradient_norm > 0.0:
                # Gradients have to be unscaled before they can be clipped.
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.gradient_norm)
            scaler.step(optimizer)
            scaler.update()
            total_step += 1

            with torch.no_grad():
                if args.etab_singlesite_loss:
                    loss, _, true_false = potts_singlesite_loss(etab, E_idx, S_true,
                                                                mask_for_loss, vocab,
                                                                from_val=True)
                else:
                    loss, _, true_false = loss_nll(S_true, log_probs, mask_for_loss)
                train_sum += torch.sum(loss * mask_for_loss).item()
                train_acc += torch.sum(true_false * mask_for_loss).item()
                train_weights += torch.sum(mask_for_loss).item()
                if nlcpl_loss is not None:
                    train_nlcpl_sum += nlcpl_loss.item()
                    train_nlcpl_batches += 1

            if (i_train_batch + 1) % 100 == 0 and train_weights > 0:
                pbar.set_postfix({
                    "loss": f"{train_sum / train_weights:.4f}",
                    "acc": f"{train_acc / train_weights:.4f}",
                })
        pbar.close()

        model.eval()
        validation_sum, validation_acc, validation_weights = 0., 0., 0.
        validation_nlcpl_sum, validation_nlcpl_batches = 0., 0

        pbar_val = tqdm(loader_valid, desc=f"Epoch {e+1} [val]", unit="batch", miniters=100)
        with torch.no_grad():
            for i_val_batch, batch in enumerate(pbar_val):
                X, S, S_true, mask, chain_M, residue_idx, chain_encoding_all, names = prepare_batch(batch, device)
                if mask.sum(dim=1).min() == 0:
                    print(f"{names} have no valid positions")
                mask_for_loss = mask * chain_M
                randn = torch.randn(chain_M.shape, device=device)

                log_probs, etab, E_idx = model(X, S, mask, chain_M, residue_idx,
                                               chain_encoding_all, randn)
                if args.etab_singlesite_loss:
                    loss, _, true_false = potts_singlesite_loss(etab, E_idx, S_true,
                                                                mask_for_loss, vocab,
                                                                from_val=True)
                else:
                    loss, _, true_false = loss_nll(S_true, log_probs, mask_for_loss)
                if args.etab_loss or args.etab_loss_only:
                    nlcpl_loss, _, n_potts_edges = nlcpl(etab, E_idx, S_true, mask)
                    if n_potts_edges > 0:
                        validation_nlcpl_sum += nlcpl_loss.item()
                        validation_nlcpl_batches += 1

                validation_sum += torch.sum(loss * mask_for_loss).item()
                validation_acc += torch.sum(true_false * mask_for_loss).item()
                validation_weights += torch.sum(mask_for_loss).item()

                if (i_val_batch + 1) % 100 == 0 and validation_weights > 0:
                    pbar_val.set_postfix({
                        "loss": f"{validation_sum / validation_weights:.4f}",
                        "acc": f"{validation_acc / validation_weights:.4f}",
                    })
        pbar_val.close()

        if train_weights == 0 or validation_weights == 0:
            raise RuntimeError("no scorable positions in this epoch; check the data paths and filters")

        train_loss = train_sum / train_weights
        train_accuracy = train_acc / train_weights
        train_perplexity = np.exp(train_loss)
        validation_loss = validation_sum / validation_weights
        validation_accuracy = validation_acc / validation_weights
        validation_perplexity = np.exp(validation_loss)

        # The combined losses are what checkpoint selection is based on, so they
        # include the Potts term whenever it is part of the objective.
        train_comb_loss = train_loss
        comb_loss = validation_loss
        train_nlcpl = validation_nlcpl = None
        if args.etab_loss or args.etab_loss_only:
            train_nlcpl = train_nlcpl_sum / max(train_nlcpl_batches, 1)
            validation_nlcpl = validation_nlcpl_sum / max(validation_nlcpl_batches, 1)
            train_comb_loss += train_nlcpl
            comb_loss += validation_nlcpl

        dt = format_float(time.time() - t0)
        summary = (f'epoch: {e+1}, step: {total_step}, time: {dt}, '
                   f'train_loss: {train_comb_loss}, val_loss: {comb_loss}, '
                   f'best_val_loss: {best_val_loss}, '
                   f'train_perp: {format_float(train_perplexity)}, '
                   f'valid_prep: {format_float(validation_perplexity)}, '
                   f'train_acc: {format_float(train_accuracy)}, '
                   f'valid_acc: {format_float(validation_accuracy)}')
        if train_nlcpl is not None:
            summary += (f'\n\ttrain_nlcpl: {format_float(train_nlcpl)}, '
                        f'valid_nlcpl: {format_float(validation_nlcpl)}')

        with open(logfile, 'a') as f:
            f.write(summary + '\n')
        print(summary)

        if comb_loss < best_val_loss:
            save_checkpoint(base_folder + 'model_weights/epoch_best.pt',
                            model, optimizer, args, e + 1, total_step, vocab)
            best_val_loss = comb_loss

        save_checkpoint(base_folder + 'model_weights/epoch_last.pt',
                        model, optimizer, args, e + 1, total_step, vocab)

        if (e + 1) % args.save_model_every_n_epochs == 0:
            save_checkpoint(base_folder + 'model_weights/epoch{}_step{}.pt'.format(e + 1, total_step),
                            model, optimizer, args, e + 1, total_step, vocab)

    # Drain the outstanding shards so the pool can shut down cleanly.
    for _ in range(len(train_futures)):
        get_one_pair(executor, train_futures, valid_futures, train_loader, valid_loader,
                     base_folder, args, requeue=False)
    executor.shutdown()


if __name__ == "__main__":
    argparser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # Data
    argparser.add_argument("--path_for_training_data", type=str, default="my_path/pdb_2021aug02", help="path for loading training data")
    argparser.add_argument("--path_for_meta_data", type=str, default="my_path/pdb_2021aug02_meta", help="path for loading meta data")
    argparser.add_argument("--path_for_outputs", type=str, default="./exp_020", help="path for logs and model weights")
    argparser.add_argument("--rescut", type=float, default=3.5, help="PDB resolution cutoff")
    argparser.add_argument("--max_protein_length", type=int, default=10000, help="maximum length of the protein complex")
    argparser.add_argument("--soluble_mpnn", type=str, default='', help='path to a csv of PDB_IDS to exclude when training a soluble version of the model')
    argparser.add_argument("--num_workers", type=int, default=12, help="number of workers to use for batch loading")
    argparser.add_argument("--num_data_workers", type=int, default=2, help="number of workers to use for parsing structures")
    argparser.add_argument("--debug", type=int, default=0, help="minimal data loading for debugging")

    # Training schedule
    argparser.add_argument("--num_epochs", type=int, default=200, help="number of epochs to train for")
    argparser.add_argument("--batch_size", type=int, default=10000, help="number of tokens for one batch")
    argparser.add_argument("--num_examples_per_epoch", type=int, default=1000000, help="number of training example to load for one epoch")
    argparser.add_argument("--reload_data_every_n_epochs", type=int, default=2, help="reload training data every n epochs")
    argparser.add_argument("--save_model_every_n_epochs", type=int, default=10, help="save model weights every n epochs")
    argparser.add_argument("--previous_checkpoint", type=str, default="", help="path for previous model weights, e.g. file.pt")
    argparser.add_argument("--load_optimizer", type=int, default=1, help="whether to load optimizer when fine-tuning a model")
    argparser.add_argument("--strict", type=int, default=1, help="enforce match between path for old model and current model")

    # Model
    argparser.add_argument("--hidden_dim", type=int, default=128, help="hidden model dimension")
    argparser.add_argument("--num_encoder_layers", type=int, default=3, help="number of encoder layers")
    argparser.add_argument("--num_decoder_layers", type=int, default=3, help="number of decoder layers")
    argparser.add_argument("--num_neighbors", type=int, default=48, help="number of neighbors for the sparse graph")
    argparser.add_argument("--dropout", type=float, default=0.1, help="dropout level; 0.0 means no dropout")
    argparser.add_argument("--backbone_noise", type=float, default=0.2, help="amount of noise added to backbone during training")
    argparser.add_argument("--output_dim", type=int, default=400, help="Potts model output dimension")

    # Optimization
    argparser.add_argument("--mixed_precision", type=int, default=1, help="train with mixed precision")
    argparser.add_argument("--gradient_norm", type=float, default=-1.0, help="clip gradient norm, set to negative to omit clipping")

    # Losses
    argparser.add_argument("--etab_loss", type=int, default=0, help="whether to add the Potts model loss to the sequence loss")
    argparser.add_argument("--etab_loss_only", type=int, default=0, help="whether to train on the Potts model loss alone")
    argparser.add_argument("--etab_singlesite_loss", type=int, default=0, help="whether to train on the single-site Potts model loss instead of the sequence loss")
    argparser.add_argument("--fixed_denom", type=float, default=2000.0, help="fixed denominator for the sequence loss, set to 0 to normalize by the number of residues")
    argparser.add_argument("--fixed_potts_denom", type=float, default=0.0, help="fixed denominator for nlcpl, set to 0 to normalize by the number of edges")

    # MSA sequences
    argparser.add_argument("--msa_seqs", type=int, default=0, help="whether to use msa sequences for sequence prediction")
    argparser.add_argument("--msa_dir", type=str, default='', help='path to MSAs')
    argparser.add_argument("--msa_batch_size", type=int, default=1, help="batch size for msa sequences")
    argparser.add_argument("--msa_match_dict", type=str, default='', help='mapping of chain ids for PDB MSAs')
    argparser.add_argument("--complex_mapping_path", type=str, default='', help='mapping of complex chain ids for PDB MSAs')
    argparser.add_argument("--consensus_seqs", type=str, default='', help="whether to use consensus sequences for sequence prediction")
    argparser.add_argument("--single_species_sample", type=int, default=0, help="whether to restrict MSA sampling to only 1 sequence per species")
    argparser.add_argument("--exclude_msa", type=str, default='', help='PDBs to exclude because of missing MSA information')
    argparser.add_argument("--id_thresh", type=float, default=0.5, help="sequence identity cutoff for msa sequences")
    argparser.add_argument("--del_thresh", type=float, default=0.2, help="deletion percent cutoff for msa sequences")
    argparser.add_argument("--insrt_thresh", type=float, default=0.2, help="insertion percent cutoff for msa sequences")
    argparser.add_argument("--remove_missing", type=int, default=1, help="whether to remove residues missing structure information")

    args = argparser.parse_args()

    args.debug = args.debug == 1
    args.load_optimizer = args.load_optimizer == 1
    args.strict = args.strict == 1
    args.mixed_precision = args.mixed_precision == 1
    args.etab_loss = args.etab_loss == 1
    args.etab_loss_only = args.etab_loss_only == 1
    args.etab_singlesite_loss = args.etab_singlesite_loss == 1
    args.msa_seqs = args.msa_seqs == 1
    args.single_species_sample = args.single_species_sample == 1
    args.remove_missing = args.remove_missing == 1

    if sum([args.etab_loss, args.etab_loss_only, args.etab_singlesite_loss]) > 1:
        argparser.error("--etab_loss, --etab_loss_only and --etab_singlesite_loss are mutually exclusive")
    if args.output_dim != 400:
        argparser.error("--output_dim must be 400; the Potts losses assume a 20x20 table per edge")

    print('starting')

    main(args)
