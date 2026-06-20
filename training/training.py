
import numpy as np
import argparse
import os.path
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

from tqdm import tqdm
import pickle
import json, time, os, sys, glob
import shutil
import warnings
import torch
from torch import optim
from torch.utils.data import DataLoader
import queue
import copy
import torch.nn as nn
import torch.nn.functional as F
import random
import os.path
import subprocess
import pandas as pd
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED, as_completed
from utils import worker_init_fn, get_pdbs, loader_pdb, build_training_clusters, PDB_dataset, StructureDataset, StructureSampler
from potts_mpnn_utils import loss_smoothed, nlcpl, loss_nll, potts_singlesite_loss, get_std_opt, ProteinMPNN

class AverageMeter:
    """Computes and stores the average and current value efficiently.
       Updates internal stats only every `update_every` calls.
    """
    def __init__(self, update_every=100):
        self.update_every = update_every
        self.reset()

    def reset(self):
        self.val = 0
        self.sum = 0
        self.count = 0
        self.avg = 0
        self._iters = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self._iters += 1

        if self._iters % self.update_every == 0:
            self.avg = self.sum / self.count if self.count > 0 else 0

def load_pickle_stream(path):
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

def start_workers(train_loader, valid_loader, base_folder, args, n_pairs=2):
    executor = ProcessPoolExecutor(max_workers=3*n_pairs, mp_context=mp.get_context('spawn'))

    train_futures = {}
    valid_futures = {}
    
    for i in range(n_pairs):
        f_train = executor.submit(get_pdbs, train_loader, i, 'train', base_folder, 1,
                                  args.max_protein_length, args.num_examples_per_epoch,
                                  args.consensus_seqs, args.msa_match_dict,
                                  args.complex_mapping_path, args.msa_dir, args.msa_seqs,
                                  args.single_species_sample, args.remove_missing,
                                  args.id_thresh, args.del_thresh, args.insrt_thresh)
        train_futures[f_train] = i

        f_valid = executor.submit(get_pdbs, valid_loader, i, 'valid', base_folder, 1,
                                  args.max_protein_length, args.num_examples_per_epoch,
                                  args.consensus_seqs, args.msa_match_dict,
                                  args.complex_mapping_path, args.msa_dir, args.msa_seqs,
                                  args.single_species_sample, args.remove_missing,
                                  args.id_thresh, args.del_thresh, args.insrt_thresh)
        valid_futures[f_valid] = i

    return executor, train_futures, valid_futures


def get_one_pair(executor, train_futures, valid_futures,
                 train_loader, valid_loader, base_folder, args, requeue=False):
    """Block until one train + one valid future are done, return results and requeue workers."""
    
    # Wait for at least one completed future from each
    done_train, _ = wait(train_futures.keys(), return_when=FIRST_COMPLETED)
    done_valid, _ = wait(valid_futures.keys(), return_when=FIRST_COMPLETED)

    # Pop one completed future each
    train_future = done_train.pop()
    valid_future = done_valid.pop()

    train_file = train_future.result()
    valid_file = valid_future.result()

    pdb_dict_train = list(load_pickle_stream(train_file))
    os.remove(train_file)
    pdb_dict_valid = list(load_pickle_stream(valid_file))
    os.remove(valid_file)

    # Requeue a new worker to replace the one consumed
    if requeue:
        # Retrieve worker index
        train_i = train_futures.pop(train_future)
        valid_i = valid_futures.pop(valid_future)
        new_train_f = executor.submit(get_pdbs, train_loader, train_i, 'train', base_folder, 1,
                                    args.max_protein_length, args.num_examples_per_epoch,
                                    args.consensus_seqs, args.msa_match_dict,
                                    args.complex_mapping_path, args.msa_dir, args.msa_seqs,
                                    args.single_species_sample, args.remove_missing,
                                    args.id_thresh, args.del_thresh, args.insrt_thresh)
        train_futures[new_train_f] = train_i

        new_valid_f = executor.submit(get_pdbs, valid_loader, valid_i, 'valid', base_folder, 1,
                                    args.max_protein_length, args.num_examples_per_epoch,
                                    args.consensus_seqs, args.msa_match_dict,
                                    args.complex_mapping_path, args.msa_dir, args.msa_seqs,
                                    args.single_species_sample, args.remove_missing,
                                    args.id_thresh, args.del_thresh, args.insrt_thresh)
        valid_futures[new_valid_f] = valid_i
    else:
        del train_futures[train_future]
        del valid_futures[valid_future]

    return pdb_dict_train, pdb_dict_valid

def get_one_pair_any(executor, train_futures, valid_futures,
                     train_loader, valid_loader, base_folder, args, requeue=False):
    """
    Return one completed train file and one completed valid file (not necessarily the same worker index).
    This collects completed futures from the union until it has one from each set.
    """
    done_train_fut = None
    done_valid_fut = None

    # Iterate over futures as they complete
    for fut in as_completed(list(train_futures.keys()) + list(valid_futures.keys())):
        if fut in train_futures and done_train_fut is None:
            done_train_fut = fut
        elif fut in valid_futures and done_valid_fut is None:
            done_valid_fut = fut

        if done_train_fut is not None and done_valid_fut is not None:
            break

    # Sanity
    if done_train_fut is None or done_valid_fut is None:
        raise RuntimeError("No completed train/valid future found — unexpected.")

    # get results (this will re-raise exceptions from worker if any)
    train_file = done_train_fut.result()
    valid_file = done_valid_fut.result()

    pdb_dict_train = list(load_pickle_stream(train_file))
    os.remove(train_file)
    pdb_dict_valid = list(load_pickle_stream(valid_file))
    os.remove(valid_file)

    # Requeue or remove from dicts (same pattern as your original code)
    if requeue:
        train_i = train_futures.pop(done_train_fut)
        valid_i = valid_futures.pop(done_valid_fut)

        new_train_f = executor.submit(get_pdbs, train_loader, train_i, 'train', base_folder, 1,
                                      args.max_protein_length, args.num_examples_per_epoch,
                                      args.consensus_seqs, args.msa_match_dict,
                                      args.complex_mapping_path, args.msa_dir, args.msa_seqs,
                                      args.single_species_sample, args.remove_missing,
                                      args.id_thresh, args.del_thresh, args.insrt_thresh)
        train_futures[new_train_f] = train_i

        new_valid_f = executor.submit(get_pdbs, valid_loader, valid_i, 'valid', base_folder, 1,
                                      args.max_protein_length, args.num_examples_per_epoch,
                                      args.consensus_seqs, args.msa_match_dict,
                                      args.complex_mapping_path, args.msa_dir, args.msa_seqs,
                                      args.single_species_sample, args.remove_missing,
                                      args.id_thresh, args.del_thresh, args.insrt_thresh)
        valid_futures[new_valid_f] = valid_i
    else:
        # remove consumed futures
        del train_futures[done_train_fut]
        del valid_futures[done_valid_fut]

    return pdb_dict_train, pdb_dict_valid


def main(args):

    scaler = torch.cuda.amp.GradScaler()
     
    device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")

    base_folder = time.strftime(args.path_for_outputs, time.localtime())

    if base_folder[-1] != '/':
        base_folder += '/'
    if not os.path.exists(base_folder):
        os.makedirs(base_folder)
    subfolders = ['model_weights']
    for subfolder in subfolders:
        if not os.path.exists(base_folder + subfolder):
            os.makedirs(base_folder + subfolder)

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
                  'pin_memory':False,
                  'num_workers': 2}

   
    if args.debug:
        args.num_examples_per_epoch = 500
        args.max_protein_length = 1000
        args.batch_size = 1
        # params["LIST"] = '/orcd/pool/003/fosterb/mpnn_data/pdb_2021aug02/debug_df2.csv'

    print('loaded args')
    print(args)
    if args.soluble_mpnn:
        exclude_df = pd.read_csv(args.soluble_mpnn)
        exclude = list(exclude_df['PDB_IDS'].values)
    else:
        exclude = []
    if args.exclude_msa:
        with open(args.exclude_msa, 'r') as f:
            exclude_msa = f.readlines()
        exclude += [cid.strip() for cid in exclude_msa]

    train, valid, test = build_training_clusters(params, args.debug, exclude)
    print('built clusters')

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

    if args.debug:
        for i, _ in enumerate(train_loader):
            pass
        for i_valid, _ in enumerate(valid_loader):
            pass
        print('debug loader len: ', i, i_valid)
    use_potts = args.etab_loss or args.etab_loss_only or args.etab_singlesite_loss
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    vocab = 21 if not args.msa_seqs else 22
    print('vocab: ', vocab)
    model = ProteinMPNN(node_features=args.hidden_dim, 
                        edge_features=args.hidden_dim, 
                        hidden_dim=args.hidden_dim, 
                        num_encoder_layers=args.num_encoder_layers, 
                        num_decoder_layers=args.num_encoder_layers, 
                        k_neighbors=args.num_neighbors, 
                        dropout=args.dropout, 
                        augment_eps=args.backbone_noise,
                        augment_type=args.noise_type,
                        augment_lim=args.augment_lim,
                        feat_type=args.feat_type,
                        use_potts=use_potts,
                        output_dim=args.output_dim,
                        node_self_sub=args.node_self_sub,
                        clone=args.clone,
                        seq_encoding=args.seq_encoding,
                        struct_predict=args.struct_predict,
                        use_struct_weights=args.use_struct_weights,
                        multimer_structure_module=args.multimer_structure_module,
                        struct_predict_pairs=args.struct_predict_pairs,
                        struct_predict_seq=args.struct_predict_seq,
                        use_seq=args.use_seq,
                        vocab=vocab,
                        num_letters=vocab,
                        gauge_normalize=args.gauge_normalize,
                        device=device
                        )
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
    kwargs = {}
    kwargs['num_workers'] = args.num_workers
    # kwargs['multiprocessing_context'] = 'spawn'
    executor, train_futures, valid_futures = start_workers(train_loader, valid_loader, base_folder, args, n_pairs=args.num_data_workers)
    print('waiting to get result')
    
    pdb_dict_train, pdb_dict_valid = get_one_pair(executor, train_futures, valid_futures,
                                          train_loader, valid_loader, base_folder, args, requeue=True)

    ## Load ESM if required
    if args.seq_encoding == 'esm2-150M':
        esm, alphabet = esmlib.pretrained.esm2_t30_150M_UR50D()
        
        esm_embed_layer = 30
        esm_embed_dim = 640
        one_hot = False
    elif args.seq_encoding == 'esm2-650M':
        esm, alphabet = esmlib.pretrained.esm2_t33_650M_UR50D()
        esm_embed_layer = 33
        esm_embed_dim = 1280
        one_hot = False
    elif args.seq_encoding == 'esm2-3B':
        esm, alphabet = esmlib.pretrained.esm2_t36_3B_UR50D()
        esm_embed_layer = 36
        esm_embed_dim = 2560
        one_hot = False
    elif args.seq_encoding == 'esm2-one_hot':
        esm, alphabet = esmlib.pretrained.esm2_t30_150M_UR50D()
        esm_embed_layer = 0
        esm_embed_dim = 22
        one_hot = True
    elif args.seq_encoding == 'one_hot':
        esm, alphabet, batch_converter, esm_embed_layer, esm_embed_dim = None, None, None, None, None
        one_hot = True
    else:
        print("seq_encoding not recognized")
        raise ValueError
    if esm is not None:
        esm = esm.to(device=device)
        batch_converter = alphabet.get_batch_converter()
        esm = esm.eval()
    print('Loaded ESM: ', args.seq_encoding)

    # raise ValueError
    
    dataset_train = StructureDataset(pdb_dict_train, truncate=None, max_length=args.max_protein_length) 
    dataset_valid = StructureDataset(pdb_dict_valid, truncate=None, max_length=args.max_protein_length)
    
    # loader_train = StructureLoader(dataset_train, batch_size=args.batch_size)
    train_batch_sampler = StructureSampler(dataset_train, batch_size=args.batch_size, device=device, flex_type=args.noise_type, augment_eps=args.backbone_noise, replicate=args.replicate,
                                            esm=esm, batch_converter=batch_converter, esm_embed_layer=esm_embed_layer, esm_embed_dim=esm_embed_dim, one_hot=one_hot,
                                            openfold_backbone=args.struct_predict, msa_seqs=args.msa_seqs, msa_batch_size=args.msa_batch_size)
    loader_train = DataLoader(dataset_train, batch_sampler=train_batch_sampler, collate_fn=train_batch_sampler.package, pin_memory=True, **kwargs)
    # loader_valid = StructureLoader(dataset_valid, batch_size=args.batch_size)
    valid_batch_sampler = StructureSampler(dataset_valid, batch_size=args.batch_size, device=device, esm=esm, batch_converter=batch_converter, esm_embed_layer=esm_embed_layer,
                                            esm_embed_dim=esm_embed_dim, one_hot=one_hot, openfold_backbone=args.struct_predict, msa_seqs=args.msa_seqs, msa_batch_size=args.msa_batch_size)
    loader_valid = DataLoader(dataset_valid, batch_sampler=valid_batch_sampler, collate_fn=valid_batch_sampler.package, pin_memory=True, **kwargs)
    reload_c = 0 
    best_val_loss = np.inf
    for e in range(args.num_epochs):
        loss_meter = AverageMeter(update_every=100)
        acc_meter = AverageMeter(update_every=100)

        t0 = time.time()
        e = epoch + e
        model.train()
        train_sum, train_weights = 0., 0.
        nlcpl_train_sum = 0.
        struct_loss_train_sum = 0.
        train_acc = 0.
        
        if e % args.reload_data_every_n_epochs == 0:
            if reload_c != 0:
                pdb_dict_train, pdb_dict_valid = get_one_pair(executor, train_futures, valid_futures,
                                          train_loader, valid_loader, base_folder, args, requeue=True)
                
                dataset_train = StructureDataset(pdb_dict_train, truncate=None, max_length=args.max_protein_length)
                train_batch_sampler = StructureSampler(dataset_train, batch_size=args.batch_size, device=device, flex_type=args.noise_type, augment_eps=args.backbone_noise, replicate=args.replicate, esm=esm, batch_converter=batch_converter,
                                                        esm_embed_layer=esm_embed_layer, esm_embed_dim=esm_embed_dim, one_hot=one_hot, openfold_backbone=args.struct_predict, msa_seqs=args.msa_seqs, msa_batch_size=args.msa_batch_size)
                loader_train = DataLoader(dataset_train, batch_sampler=train_batch_sampler, collate_fn=train_batch_sampler.package, pin_memory=True, **kwargs)
                dataset_valid = StructureDataset(pdb_dict_valid, truncate=None, max_length=args.max_protein_length)
                valid_batch_sampler = StructureSampler(dataset_valid, batch_size=args.batch_size, device=device, esm=esm, batch_converter=batch_converter, esm_embed_layer=esm_embed_layer, esm_embed_dim=esm_embed_dim, one_hot=one_hot,
                                                        openfold_backbone=args.struct_predict, msa_seqs=args.msa_seqs, msa_batch_size=args.msa_batch_size)
                loader_valid = DataLoader(dataset_valid, batch_sampler=valid_batch_sampler, collate_fn=valid_batch_sampler.package, pin_memory=True, **kwargs)

            reload_c += 1
        train_batch_sampler._set_epoch(e)
        valid_batch_sampler._set_epoch(e)
        pbar = tqdm(total=len(loader_train), desc=f"Epoch {e+1} [train]", unit="batch", miniters=100)
        for i_train_batch, batch in enumerate(loader_train):
            X, S, S_true, mask, lengths, chain_M, residue_idx, mask_self, chain_encoding_all, all_chain_lens, backbone_4x4, names = batch
            if mask.sum(dim=1).min() == 0:
                print(f"{names} have no valid positions")
                print(mask.sum(dim=1))
            residue_idx = residue_idx.to(device=device)
            S = S.to(device=device)
            S_true = S_true.to(device=device)
            X = X.to(device=device)
            mask = mask.to(device=device)
            mask_self = mask_self.to(device=device)
            chain_M = chain_M.to(device=device)
            chain_encoding_all =chain_encoding_all.to(device=device)
            optimizer.zero_grad()
            mask_for_loss = mask*chain_M
            
            if args.mixed_precision:
                with torch.cuda.amp.autocast():
                    log_probs, etab, E_idx, frames, positions = model(X, S, mask, chain_M, residue_idx, chain_encoding_all, all_chain_lens, e, replicate=args.replicate)
                    _, loss_av_smoothed = loss_smoothed(S_true, log_probs, mask_for_loss, vocab, fixed_denom=args.fixed_denom)
                    if args.etab_loss:
                        nlcpl_loss, nlcpl_count = nlcpl(etab, E_idx, S_true, mask, fixed_denom=args.fixed_potts_denom)
                        if nlcpl_count > 0: loss_av_smoothed += nlcpl_loss
                        elif nlcpl_count < 0:
                            print(names)
                            print(log_probs.isnan().any(), print(etab.isnan().any()))
                            raise ValueError
                        else:
                            continue
                    elif args.etab_loss_only:
                        loss_av_smoothed, nlcpl_count = nlcpl(etab, E_idx, S_true, mask, fixed_denom=args.fixed_potts_denom)
                        nlcpl_loss = loss_av_smoothed
                    elif args.etab_singlesite_loss:
                        _, loss_av_smoothed = potts_singlesite_loss(etab, E_idx, S_true, mask, vocab, from_val=False)
                    if args.struct_predict:
                        backbone_4x4 = backbone_4x4.to(device=device)
                        struct_loss, _, struct_success = structure_loss(frames, backbone_4x4, mask)
                        if struct_success >= 0: loss_av_smoothed += struct_loss
                        else:
                            print(names)
                            print(log_probs.isnan().any(), etab.isnan().any())
                            print(etab.isnan().any())
                            print(backbone_4x4.isnan().any())
                            for name, p in model.named_parameters():
                                if p.isnan().any():
                                    print(name)
                            print('done checking parameters')
                if ((not args.etab_loss or not args.etab_loss_only) or nlcpl_count >= 0) and (not args.struct_predict or struct_success >= 0):
                    scaler.scale(loss_av_smoothed).backward()
                    
                if args.gradient_norm > 0.0:
                    total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.gradient_norm)

                scaler.step(optimizer)
                scaler.update()
            else:
                log_probs, etab, E_idx, frames, positions = model(X, S, mask, chain_M, residue_idx, chain_encoding_all, all_chain_lens, e, replicate=args.replicate)
                _, loss_av_smoothed = loss_smoothed(S_true, log_probs, mask_for_loss, vocab, fixed_denom=args.fixed_denom)
                if args.etab_loss:
                        nlcpl_loss, nlcpl_count = nlcpl(etab, E_idx, S_true, mask, fixed_denom=args.fixed_potts_denom)
                        if nlcpl_count > 0: loss_av_smoothed += nlcpl_loss
                        elif nlcpl_count < 0:
                            print(names)
                            print(log_probs.isnan().any(), print(etab.isnan().any()))
                            raise ValueError
                        else:
                            continue
                elif args.etab_loss_only:
                        loss_av_smoothed, nlcpl_count = nlcpl(etab, E_idx, S_true, mask, fixed_denom=args.fixed_potts_denom)
                        nlcpl_loss = loss_av_smoothed
                elif args.etab_singlesite_loss:
                    _, loss_av_smoothed = potts_singlesite_loss(etab, E_idx, S_true, mask, vocab, from_val=False)
                if args.struct_predict:
                        backbone_4x4 = backbone_4x4.to(device=device)
                        struct_loss, _, struct_success = structure_loss(frames, backbone_4x4, mask)
                        if struct_success >= 0: loss_av_smoothed += struct_loss
                        else:
                            print(names)
                            print(log_probs.isnan().any(), etab.isnan().any())
                            print(etab.isnan().any())
                            print(backbone_4x4.isnan().any())
                            for name, p in model.named_parameters():
                                if p.isnan().any():
                                    print(name)
                            print('done checking parameters')
                if args.etab_loss and nlcpl_count >= 0:
                    loss_av_smoothed.backward()

                if args.gradient_norm > 0.0:
                    total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.gradient_norm)

                optimizer.step()
            
            if args.etab_singlesite_loss:
                loss, loss_av, true_false = potts_singlesite_loss(etab, E_idx, S_true, mask, vocab, from_val=True)
            else:
                loss, loss_av, true_false = loss_nll(S_true, log_probs, mask_for_loss)
            if args.etab_loss or args.etab_loss_only:
                # nlcpl_loss, nlcpl_count = nlcpl(etab, E_idx, S_true, mask)
                if nlcpl_count >= 0:
                    nlcpl_train_sum += nlcpl_loss.cpu().item()
                else:
                    nlcpl_train_sum += np.mean(nlcpl_train_sum)
            # train_sum += torch.sum(loss * mask_for_loss).cpu().data.numpy()
            # train_acc += torch.sum(true_false * mask_for_loss).cpu().data.numpy()
            # train_weights += torch.sum(mask_for_loss).cpu().data.numpy()
            if args.struct_predict:
                struct_loss_train_sum += struct_loss.cpu().item()

            total_step += 1
            # per-example statistics
            batch_loss = torch.sum(loss * mask_for_loss).cpu().data.numpy()
            batch_acc = torch.sum(true_false * mask_for_loss).cpu().data.numpy()
            batch_weight = torch.sum(mask_for_loss).cpu().data.numpy()
            train_sum += batch_loss
            train_acc += batch_acc
            train_weights += batch_weight

            # update meters
            loss_meter.update(batch_loss / (batch_weight + 1e-8), batch_weight)
            acc_meter.update(batch_acc / (batch_weight + 1e-8), batch_weight)

            # update tqdm bar
            if (i_train_batch + 1) % 100 == 0:
                pbar.update(100)
                pbar.set_postfix({
                    "loss": f"{loss_meter.avg:.4f}",
                    "acc": f"{acc_meter.avg:.4f}"
                })

        model.eval()
        val_loss_meter = AverageMeter(update_every=100)
        val_acc_meter = AverageMeter(update_every=100)
        pbar_val = tqdm(total=len(loader_valid), desc=f"Epoch {e+1} [val]", unit="batch", miniters=100)
        with torch.no_grad():
            validation_sum, validation_weights = 0., 0.
            nlcpl_validation_sum = 0.
            struct_loss_val_sum = 0.
            validation_acc = 0.
            for i_val_batch, batch in enumerate(loader_valid):
                X, S, S_true, mask, lengths, chain_M, residue_idx, mask_self, chain_encoding_all, all_chain_lens, backbone_4x4, names = batch
                if mask.sum(dim=1).min() == 0:
                    print(f"{names} have no valid positions")
                    print(mask.sum(dim=1))
                residue_idx = residue_idx.to(device=device)
                S = S.to(device=device)
                S_true = S_true.to(device=device)
                X = X.to(device=device)
                mask = mask.to(device=device)
                mask_self = mask_self.to(device=device)
                chain_M = chain_M.to(device=device)
                chain_encoding_all =chain_encoding_all.to(device=device)
                log_probs, etab, E_idx, frames, positions = model(X, S, mask, chain_M, residue_idx, chain_encoding_all)
                mask_for_loss = mask*chain_M
                loss, loss_av, true_false = loss_nll(S_true, log_probs, mask_for_loss)
                if args.etab_loss or args.etab_loss_only:
                    nlcpl_loss, nlcpl_count = nlcpl(etab, E_idx, S_true, mask)
                    if nlcpl_count > 0:
                        nlcpl_validation_sum += nlcpl_loss.cpu().item()
                    else:
                        nlcpl_validation_sum += np.mean(nlcpl_validation_sum)
                elif args.etab_singlesite_loss:
                    loss, loss_av, true_false = potts_singlesite_loss(etab, E_idx, S_true, mask_for_loss, vocab, from_val=True)
                if args.struct_predict:
                    backbone_4x4 = backbone_4x4.to(device=device)
                    struct_loss, _, struct_success = structure_loss(frames, backbone_4x4, mask)
                    if struct_success >= 0:
                        struct_loss_val_sum += struct_loss.cpu().item()
                    else:
                        struct_loss_val_sum += np.mean(struct_loss_val_sum)
                
                # validation_sum += torch.sum(loss * mask_for_loss).cpu().data.numpy()
                # validation_acc += torch.sum(true_false * mask_for_loss).cpu().data.numpy()
                # validation_weights += torch.sum(mask_for_loss).cpu().data.numpy()

                batch_loss = torch.sum(loss * mask_for_loss).cpu().data.numpy()
                batch_acc = torch.sum(true_false * mask_for_loss).cpu().data.numpy()
                batch_weight = torch.sum(mask_for_loss).cpu().data.numpy()
                validation_sum += batch_loss
                validation_acc += batch_acc
                validation_weights += batch_weight

                val_loss_meter.update(batch_loss / (batch_weight + 1e-8), batch_weight)
                val_acc_meter.update(batch_acc / (batch_weight + 1e-8), batch_weight)

                if (i_val_batch + 1) % 100 == 0:
                    pbar_val.update(100)
                    pbar_val.set_postfix({
                        "loss": f"{val_loss_meter.avg:.4f}",
                        "acc": f"{val_acc_meter.avg:.4f}"
                    })
        
        train_loss = train_sum / train_weights
        train_comb_loss = copy.deepcopy(train_loss)
        train_accuracy = train_acc / train_weights
        train_perplexity = np.exp(train_loss)
        validation_loss = validation_sum / validation_weights
        validation_accuracy = validation_acc / validation_weights
        validation_perplexity = np.exp(validation_loss)
        
        train_perplexity_ = np.format_float_positional(np.float32(train_perplexity), unique=False, precision=3)     
        validation_perplexity_ = np.format_float_positional(np.float32(validation_perplexity), unique=False, precision=3)
        train_accuracy_ = np.format_float_positional(np.float32(train_accuracy), unique=False, precision=3)
        validation_accuracy_ = np.format_float_positional(np.float32(validation_accuracy), unique=False, precision=3)
        comb_loss = copy.deepcopy(validation_loss)

        if args.etab_loss or args.etab_loss_only:
            train_nlcpl = nlcpl_train_sum / (i_train_batch + 1)
            validation_nlcpl = nlcpl_validation_sum / (i_val_batch + 1)
            train_comb_loss += train_nlcpl
            comb_loss += validation_nlcpl
            train_nlcpl = np.format_float_positional(np.float32(train_nlcpl), unique=False, precision=3)   
            validation_nlcpl = np.format_float_positional(np.float32(validation_nlcpl), unique=False, precision=3)   
        if args.struct_predict:
            train_struct_loss = struct_loss_train_sum / (i_train_batch + 1)
            val_struct_loss = struct_loss_val_sum / (i_val_batch + 1)
            train_comb_loss += train_struct_loss
            comb_loss += val_struct_loss
            train_struct_loss = np.format_float_positional(np.float32(train_struct_loss), unique=False, precision=3)   
            val_struct_loss = np.format_float_positional(np.float32(val_struct_loss), unique=False, precision=3) 
                

        t1 = time.time()
        dt = np.format_float_positional(np.float32(t1-t0), unique=False, precision=1) 
        with open(logfile, 'a') as f:
            f.write(f'epoch: {e+1}, step: {total_step}, time: {dt}, train_loss: {train_comb_loss}, val_loss: {comb_loss}, best_val_loss: {best_val_loss}, train_perp: {train_perplexity_}, valid_prep: {validation_perplexity_}, train_acc: {train_accuracy_}, valid_acc: {validation_accuracy_}\n')
            if args.etab_loss or args.etab_loss_only:
                f.write(f'\ttrain_nlcpl: {train_nlcpl}, valid_nlcpl: {validation_nlcpl}\n')
            if args.struct_predict:
                f.write(f'\ttrain_struct_loss: {train_struct_loss}, valid_struct_loss: {val_struct_loss}\n')

        print(f'epoch: {e+1}, step: {total_step}, time: {dt}, train_loss: {train_comb_loss}, val_loss: {comb_loss}, best_val_loss: {best_val_loss}, train_perp: {train_perplexity_}, valid_prep: {validation_perplexity_}, train_acc: {train_accuracy_}, valid_acc: {validation_accuracy_}')
        if args.etab_loss or args.etab_loss_only:
            print(f'\ttrain_nlcpl: {train_nlcpl}, valid_nlcpl: {validation_nlcpl}')
        if args.struct_predict:
            print(f'\ttrain_struct_loss: {train_struct_loss}, valid_struct_loss: {val_struct_loss}\n')

        if comb_loss < best_val_loss:
            checkpoint_filename_last = base_folder+'model_weights/epoch_best.pt'.format(e+1, total_step)
            torch.save({
                    'epoch': e+1,
                    'step': total_step,
                    'num_edges' : args.num_neighbors,
                    'noise_level': args.backbone_noise,
                    'noise_type': args.noise_type,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.optimizer.state_dict(),
                    }, checkpoint_filename_last)
            best_val_loss = comb_loss
        
        checkpoint_filename_last = base_folder+'model_weights/epoch_last.pt'.format(e+1, total_step)
        torch.save({
                    'epoch': e+1,
                    'step': total_step,
                    'num_edges' : args.num_neighbors,
                    'noise_level': args.backbone_noise,
                    'noise_type': args.noise_type,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.optimizer.state_dict(),
                    }, checkpoint_filename_last)

        if (e+1) % args.save_model_every_n_epochs == 0:
            checkpoint_filename = base_folder+'model_weights/epoch{}_step{}.pt'.format(e+1, total_step)
            torch.save({
                    'epoch': e+1,
                    'step': total_step,
                    'num_edges' : args.num_neighbors,
                    'noise_level': args.backbone_noise,
                    'noise_type': args.noise_type, 
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.optimizer.state_dict(),
                    }, checkpoint_filename)
    num_futs = len(train_futures)
    for _ in range(num_futs):
        _, _ = get_one_pair(executor, train_futures, valid_futures, train_loader, valid_loader, base_folder, args, requeue=False)
                
if __name__ == "__main__":
    argparser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    argparser.add_argument("--path_for_training_data", type=str, default="my_path/pdb_2021aug02", help="path for loading training data") 
    argparser.add_argument("--path_for_meta_data", type=str, default="my_path/pdb_2021aug02_meta", help="path for loading meta data")
    argparser.add_argument("--path_for_outputs", type=str, default="./exp_020", help="path for logs and model weights")
    argparser.add_argument("--previous_checkpoint", type=str, default="", help="path for previous model weights, e.g. file.pt")
    argparser.add_argument("--num_epochs", type=int, default=200, help="number of epochs to train for")
    argparser.add_argument("--save_model_every_n_epochs", type=int, default=10, help="save model weights every n epochs")
    argparser.add_argument("--reload_data_every_n_epochs", type=int, default=2, help="reload training data every n epochs")
    argparser.add_argument("--num_examples_per_epoch", type=int, default=1000000, help="number of training example to load for one epoch")
    argparser.add_argument("--batch_size", type=int, default=10000, help="number of tokens for one batch")
    argparser.add_argument("--max_protein_length", type=int, default=10000, help="maximum length of the protein complext")
    argparser.add_argument("--hidden_dim", type=int, default=128, help="hidden model dimension")
    argparser.add_argument("--num_encoder_layers", type=int, default=3, help="number of encoder layers") 
    argparser.add_argument("--num_decoder_layers", type=int, default=3, help="number of decoder layers")
    argparser.add_argument("--num_neighbors", type=int, default=48, help="number of neighbors for the sparse graph")   
    argparser.add_argument("--dropout", type=float, default=0.1, help="dropout level; 0.0 means no dropout")
    argparser.add_argument("--backbone_noise", type=float, default=0.2, help="amount of noise added to backbone during training")
    argparser.add_argument("--noise_type", type=str, default='atomic', help="type of noise to add during training")
    argparser.add_argument("--augment_lim", type=float, default=1.0, help='RMSD limit for noise')
    argparser.add_argument("--rescut", type=float, default=3.5, help="PDB resolution cutoff")
    argparser.add_argument("--debug", type=bool, default=False, help="minimal data loading for debugging")
    argparser.add_argument("--gradient_norm", type=float, default=-1.0, help="clip gradient norm, set to negative to omit clipping")
    argparser.add_argument("--mixed_precision", type=int, default=1, help="train with mixed precision")
    argparser.add_argument("--replicate", type=int, default=1, help='replicate to use in setting seed for noise')
    argparser.add_argument("--feat_type", type=str, default="protein_mpnn", help="type of featurizer to use")
    argparser.add_argument("--fixed_denom", type=float, default=2000.0, help="fixed denominator for loss_smoothed")
    argparser.add_argument("--fixed_potts_denom", type=float, default=0.0, help="fixed denominator for nlcpl")
    argparser.add_argument("--etab_loss", type=int, default=0, help="whether to use Potts model loss")
    argparser.add_argument("--etab_singlesite_loss", type=int, default=0, help="whether to use singlesite Potts model loss")
    argparser.add_argument("--etab_loss_only", type=int, default=0, help="whether to only use Potts model loss")
    argparser.add_argument("--output_dim", type=int, default=400, help="Potts model output dimension")
    argparser.add_argument("--node_self_sub", type=str, default=None, help="whether to use sequence probabilities as Potts model self energies")
    argparser.add_argument("--clone", type=int, default=1, help="whether to clone node tensors if using for Potts model self energies")
    argparser.add_argument("--seq_encoding", type=str, default="one_hot", help="Sequence encoding to use")
    argparser.add_argument("--num_workers", type=int, default=12, help="number of workers to use for data loading")
    argparser.add_argument("--struct_predict", type=int, default=0, help="whether to use end-to-end structure supervision")
    argparser.add_argument("--use_struct_weights", type=int, default=0, help="whether to use pre-trained weights for end-to-end structure supervision")
    argparser.add_argument("--multimer_structure_module", type=int, default=0, help="whether to use multimer model for end-to-end structure supervision")
    argparser.add_argument("--struct_predict_pairs", type=int, default=1, help="whether to use pairs for end-to-end structure supervision")
    argparser.add_argument("--struct_predict_seq", type=int, default=1, help="whether to use sequence for end-to-end structure supervision")
    argparser.add_argument("--load_optimizer", type=int, default=1, help="whether to load optimizer when fine-tuning a model")
    argparser.add_argument("--strict", type=int, default=1, help="enforce match between path for old model and current model")
    argparser.add_argument('--use_seq', type=int, default=1, help='whether to use sequence info in autoregressive decoding')
    argparser.add_argument("--consensus_seqs", type=str, default='', help="whether to use consensus sequences for sequence prediction")
    argparser.add_argument("--msa_seqs", type=int, default=0, help="whether to use msa sequences for sequence prediction")
    argparser.add_argument("--single_species_sample", type=int, default=0, help="whether to restrict MSA sampling to only 1 sequence per species")
    argparser.add_argument("--msa_dir", type=str, default='', help='path to MSAs')
    argparser.add_argument("--msa_match_dict", type=str, default='', help='mapping of chain ids for PDB MSAs')
    argparser.add_argument("--complex_mapping_path", type=str, default='', help='mapping of complex chain ids for PDB MSAs')
    argparser.add_argument("--msa_batch_size", type=int, default=1, help="batch size for msa sequences")
    argparser.add_argument("--exclude_msa", type=str, default='', help='PDBs to exclude because of missing MSA information')
    argparser.add_argument("--id_thresh", type=float, default=0.5, help="sequence identity cutoff for msa sequences")
    argparser.add_argument("--del_thresh", type=float, default=0.2, help="deletion percent cutoff for msa sequences")
    argparser.add_argument("--insrt_thresh", type=float, default=0.2, help="insertion percent cutoff for msa sequences")
    argparser.add_argument("--remove_missing", type=int, default=1, help="whether to remove residues missing structure information")
    argparser.add_argument("--soluble_mpnn", type=str, default='', help='path for pdb_ids to exclude when training a soluble version of the model')
    argparser.add_argument("--num_data_workers", type=int, default=2, help="number of workers to use for processing data")
    argparser.add_argument("--gauge_normalize", type=int, default=0, help="whether to gauge normalize energy table")
    args = argparser.parse_args()

    args.etab_loss = args.etab_loss == 1
    args.etab_singlesite_loss = args.etab_singlesite_loss == 1
    args.etab_loss_only = args.etab_loss_only == 1
    args.struct_predict = args.struct_predict == 1
    args.use_struct_weights = args.use_struct_weights == 1
    args.multimer_structure_module = args.multimer_structure_module == 1
    args.struct_predict_pairs = args.struct_predict_pairs == 1
    args.struct_predict_seq = args.struct_predict_seq == 1
    args.load_optimizer = args.load_optimizer == 1
    args.strict = args.strict == 1
    args.clone = args.clone == 1
    args.use_seq = args.use_seq == 1
    args.msa_seqs = args.msa_seqs == 1
    args.single_species_sample = args.single_species_sample == 1
    args.remove_missing = args.remove_missing == 1
    args.mixed_precision = args.mixed_precision == 1
    args.gauge_normalize = args.gauge_normalize == 1

    print('starting')    

    main(args)   
