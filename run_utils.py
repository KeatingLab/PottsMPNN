import torch
import torch.nn.functional as F
import copy
import numpy as np
import os
import json
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Rectangle
from matplotlib.collections import LineCollection
from collections import defaultdict
from Bio.PDB import PDBParser, PDBIO
from Bio.Data.IUPACData import protein_letters_1to3
import etab_utils as etab_utils
from potts_mpnn_utils import parse_PDB_seq_only, tied_featurize

def gather_nodes(nodes, neighbor_idx):
    """
    Gather neighbor node features for each node in a batch.

    This helper converts node features shaped `[B, N, C]` and neighbor indices
    shaped `[B, N, K]` into neighbor features shaped `[B, N, K, C]`.

    Parameters
    ----------
    nodes : torch.Tensor, shape (B, N, C)
        Node feature tensor where B=batch, N=num_nodes, C=channels.
    neighbor_idx : torch.Tensor, shape (B, N, K)
        Integer indices of neighbors for each node.

    Returns
    -------
    torch.Tensor, shape (B, N, K, C)
        Gathered neighbor features.
    """
    # Flatten neighbor indices per batch: [B, N, K] -> [B, N*K]
    neighbors_flat = neighbor_idx.view((neighbor_idx.shape[0], -1))
    # Expand index to select all feature channels: [B, N*K, C]
    neighbors_flat = neighbors_flat.unsqueeze(-1).expand(-1, -1, nodes.size(2))
    # Gather across node dimension and reshape back to [B, N, K, C]
    neighbor_features = torch.gather(nodes, 1, neighbors_flat)
    neighbor_features = neighbor_features.view(list(neighbor_idx.shape)[:3] + [-1])
    return neighbor_features

def cat_neighbors_nodes(h_nodes, h_neighbors, E_idx):
    """
    Concatenate edge/neighbour features with gathered node features.

    This is a small convenience wrapper that gathers node features for the
    neighbor indices `E_idx` and concatenates them with precomputed
    `h_neighbors` along the last dimension.
    """
    h_nodes = gather_nodes(h_nodes, E_idx)
    h_nn = torch.cat([h_neighbors, h_nodes], -1)
    return h_nn

def optimize_sequence(seq, etab, E_idx, mask, chain_mask, opt_type, seq_encoder, model=None, h_E=None, h_EXV_encoder=None, h_V=None, constant=None, decoding_order=None):
    """
    Sequence optimization wrapper supporting several strategies.

    Parameters
    ----------
    seq : list or str
        Input sequence in a human-readable form (converted by `seq_encoder`).
    etab, E_idx : torch.Tensor
        Energy table and neighbor index tensors used by energy calculations.
    mask, chain_mask : torch.Tensor
        Binary masks indicating valid positions and chain membership.
    opt_type : str
        Optimization strategy indicator (e.g., contains 'nodes' or 'converge').
    seq_encoder : callable
        Function that encodes sequences to integer tensors.
    ener_fn : callable
        Energy evaluation function used in some modes (returns predicted_E).
    ... (other arguments forwarded to specialized flows)

    Returns
    -------
    torch.Tensor or list
        Best sequence (encoded or as characters depending on branch).
    """
    etab = etab.clone().view(etab.shape[0], etab.shape[1], etab.shape[2], int(np.sqrt(etab.shape[3])), int(np.sqrt(etab.shape[3])))
    etab = torch.nn.functional.pad(etab, (0, 2, 0, 2), "constant", 0)
    seq = torch.Tensor(seq_encoder(seq)).unsqueeze(0).to(dtype=torch.int64, device=E_idx.device)
    
    if decoding_order is None:
        decoding_order = np.arange(seq.shape[1])

    if 'nodes' not in opt_type:
        ener_delta = 1
        iters_done = 0
        if 'converge' not in opt_type:
            max_iters = 1
        else:
            max_iters = 1000
        while (ener_delta != 0 and iters_done < max_iters):
            ener_delta = 0
            for pos in decoding_order:
                if not mask[0,pos] or not chain_mask[0,pos]:
                    continue
                sort_seqs = [] 
                
                for mut_ind in range(20):
                    mut_seq = copy.deepcopy(seq)
                    mut_seq[0, pos] = mut_ind
                    sort_seqs.append(mut_seq)

                sort_seqs = torch.stack(sort_seqs, dim=1).to(etab.device)

                predicted_E = etab_utils.positional_potts_energy(etab, E_idx, seq, pos)
                seq = sort_seqs[0, predicted_E.argmin()].unsqueeze(0)
                ener_delta += torch.min(predicted_E)

            iters_done += 1
        best_seq = seq[0]
    else:
        S = seq.clone() # [B, L]
        h_S = model.W_s(S)             # [B, L, D]
        h_V_stack = [h_V] + [torch.zeros_like(h_V, device=h_V.device) for _ in range(len(model.decoder_layers))]
        log_probs = []

        # Ensure decoding order has batch dimension
        decoding_order = torch.as_tensor(decoding_order, dtype=torch.long, device=h_V.device).unsqueeze(0)
        mask_1D = mask.view([mask.size(0), mask.size(1), 1, 1])
        mask_bw = torch.ones((mask.size(0), mask.size(1), E_idx.size(2), 1)) * mask_1D
        mask_bw[:,:,0,0] = 0
        mask_fw = mask_1D * (1 - mask_bw)
        h_EXV_encoder_fw = h_EXV_encoder * mask_fw

        for t_ in range(seq.shape[1]):
            t = decoding_order[:, t_]  # [B]
            mask_gathered = torch.gather(mask, 1, t[:, None])  # [B, 1]

            if (mask_gathered == 0).all():
                continue

            # --- MASK the current position ---
            h_S_masked = h_S.clone()
            # Expand t to match h_S shape for scatter
            # index = t[:, None, None].expand(-1, 1, h_S.shape[-1])  # [B, 1, D]
            h_EXV_encoder_t = torch.gather(
                h_EXV_encoder_fw,
                1,
                t[:, None, None, None].expand(-1, 1, h_EXV_encoder_fw.shape[-2], h_EXV_encoder_fw.shape[-1]),
            )

            # Hidden layers
            E_idx_t = torch.gather(E_idx, 1, t[:, None, None].expand(-1, 1, E_idx.shape[-1]))
            h_E_t = torch.gather(
                h_E, 1, t[:, None, None, None].expand(-1, 1, h_E.shape[-2], h_E.shape[-1])
            )
            h_ES_t = cat_neighbors_nodes(h_S_masked, h_E_t, E_idx_t)
            
            mask_t = torch.gather(mask, 1, t[:, None])

            for l, layer in enumerate(model.decoder_layers):
                h_ESV_decoder_t = cat_neighbors_nodes(h_V_stack[l], h_ES_t, E_idx_t)
                # h_ESV_decoder_t[:,:,0] = h_EXV_encoder_t[:,:,0]
                h_V_t = torch.gather(
                    h_V_stack[l], 1, t[:, None, None].expand(-1, 1, h_V_stack[l].shape[-1])
                )
                
                h_ESV_t = (
                    torch.gather(
                        mask_bw,
                        1,
                        t[:, None, None, None].expand(-1, 1, mask_bw.shape[-2], mask_bw.shape[-1]),
                    )
                    * h_ESV_decoder_t
                    + h_EXV_encoder_t
                )
                h_V_stack[l + 1].scatter_(
                    1,
                    t[:, None, None].expand(-1, 1, h_V.shape[-1]),
                    layer(h_V_t, h_ESV_t, mask_V=mask_t),
                )

            # Compute residue probabilities
            h_V_t = torch.gather(
                h_V_stack[-1], 1, t[:, None, None].expand(-1, 1, h_V_stack[-1].shape[-1])
            )[:, 0]
            logits = model.W_out(h_V_t)
            probs = F.softmax(logits - constant[None, :] * 1e8, dim=-1)

            # Choose best residue (argmax)
            S_t = torch.argmax(probs, dim=-1, keepdim=True)  # [B, 1]

            # Log-probability of chosen residue
            log_p_t = torch.log(torch.gather(probs, 1, S_t) + 1e-8).mean()
            log_probs.append(log_p_t.item())

            # Update sequence embedding at this position
            temp1 = model.W_s(S_t)  # [B, 1, D]
            h_S.scatter_(1, t[:, None, None].expand(-1, 1, temp1.shape[-1]), temp1)
            S.scatter_(1, t[:, None], S_t)
    
        best_seq = S[0]
    return best_seq

def string_to_int(s):
    """
    Convert a string to an integer by summing character values.

    Parameters
    ----------
    s : str
        Input string to convert.

    Returns
    -------
    result : int
        Integer representation of the string.
    """
    result = 0
    for char in s:
        value = ord(char.lower()) - ord('a')  # a=0, b=1, ..., z=25
        result += value
    return result

def process_configs(cfg):
    """
    Loads input JSONL files for inference configuration.

    Parameters
    ----------
    cfg : OmegaConf object
        Configuration object with paths to JSONL files.

    Returns
    -------
    list of configuration dicts or None
    """
    if os.path.isfile(cfg.inference.fixed_positions_json):
        with open(cfg.inference.fixed_positions_json, 'r') as json_file:
            json_list = list(json_file)
        for json_str in json_list:
            fixed_positions_dict = json.loads(json_str)
    else:
        fixed_positions_dict = None
    
    if os.path.isfile(cfg.inference.pssm_json):
        with open(cfg.inference.pssm_json, 'r') as json_file:
            json_list = list(json_file)
        pssm_dict = {}
        for json_str in json_list:
            pssm_dict.update(json.loads(json_str))
    else:
        pssm_dict = None
    
    
    if os.path.isfile(cfg.inference.omit_AA_json):
        with open(cfg.inference.omit_AA_json, 'r') as json_file:
            json_list = list(json_file)
        for json_str in json_list:
            omit_AA_dict = json.loads(json_str)
    else:
        omit_AA_dict = None
    
    
    if os.path.isfile(cfg.inference.bias_AA_json):
        with open(cfg.inference.bias_AA_json, 'r') as json_file:
            json_list = list(json_file)
        for json_str in json_list:
            bias_AA_dict = json.loads(json_str)
    else:
        bias_AA_dict = None
    
    
    if os.path.isfile(cfg.inference.tied_positions_json):
        with open(cfg.inference.tied_positions_json, 'r') as json_file:
            json_list = list(json_file)
        for json_str in json_list:
            tied_positions_dict = json.loads(json_str)
    else:
        tied_positions_dict = None

    if os.path.isfile(cfg.inference.bias_by_res_json):
        with open(cfg.inference.bias_by_res_json, 'r') as json_file:
            json_list = list(json_file)
    
        for json_str in json_list:
            bias_by_res_dict = json.loads(json_str)
    else:
        bias_by_res_dict = None

    omit_AAs_list = cfg.inference.omit_AAs
    alphabet = 'ACDEFGHIKLMNPQRSTVWYX-'
    if cfg.model.vocab == 21:
        alphabet = alphabet[:-1]
    omit_AAs_np = np.array([AA in omit_AAs_list for AA in alphabet]).astype(np.float32)

    return fixed_positions_dict, pssm_dict, omit_AA_dict, bias_AA_dict, tied_positions_dict, bias_by_res_dict, omit_AAs_np

def is_float(s):
    """
    Checks if the input string 's' can be converted to a float.
    Returns the converted float if it can, None otherwise.
    """
    try:
        return float(s)
    except ValueError:
        return None

def process_data(cfg, binding_energy_chains):
    """
    Process data settings for energy prediction.

    Parameters
    ----------
    cfg : OmegaConf object
    binding_energy_chains : None or dict of chain list pairs for binding energy calculation

    Returns
    -------
    dataset_settings : dict of dicts
        Processed dataset settings per pdb
    """
    # Load pdb information
    with open(cfg.input_list, 'r') as f:
        pdb_list = f.readlines()
    pdb_list = [line.strip() for line in pdb_list]
    mutant_data = {'pdb': [], 'sequences': [], 'partitioned_sequences': [], 'ddG_expt': [], 'mut_chains': []}
    chain_lens_dicts = {}
    mut_alphabet = 'ACDEFGHIKLMNPQRSTVWY'
    # Load mutant sequence information
    if cfg.mutant_fasta is not None: # Predict energies for provided mutant sequences from a FASTA file
        with open(cfg.mutant_fasta, 'r') as f:
            mutant_seq_lines = f.readlines()
        mutant_seqs = defaultdict(list)
        for pdb, line in zip(mutant_seq_lines[::2], mutant_seq_lines[1::2]):
            mutant_seqs[pdb.strip().split('|')[0].strip('>')].append((pdb.strip(), line.strip()))

        for pdb in pdb_list:
            # Gather information about wild-type sequence
            wt_info = parse_PDB_seq_only(os.path.join(cfg.input_dir, pdb + '.pdb'), skip_gaps=cfg.inference.skip_gaps)
            for header, seq in mutant_seqs[pdb]:
                header = header.strip('>')
                # Parse mutant sequence header
                header_parts = header.split('|')
                assert len(header_parts) <= 3, "Header information cannot exceed 3 '|' parts"
                mut_chains = None
                ddG_expt = None
                if len(header_parts) == 2:
                    ddG_expt = is_float(header_parts[1])
                    if not ddG_expt:
                        mut_chains = header_parts[1]
                elif len(header_parts) == 3:
                    mut_chains = header_parts[1]
                    ddG_expt = is_float(header_parts[2])
                if not ddG_expt: ddG_expt = np.nan

                # Create full mutant sequence
                mut_seq = []
                if mut_chains: # If chain info in header, processes provided sequence accordingly
                    mut_chains = mut_chains.split(':')
                else: # Assume mutant sequence provided has all chains present
                    assert len(wt_info['chain_order']) == len(seq.split(':')), "If chains not specified, mutant sequence must contain information on all chains"
                    mut_chains = wt_info['chain_order']
                mut_seq_dict = {chain: chain_seq for chain, chain_seq in zip(mut_chains, seq.split(':'))}

                for chain in wt_info['chain_order']:
                    if chain in mut_seq_dict: # Use mutant sequence
                        assert len(mut_seq_dict[chain]) == len(wt_info[f'seq_chain_{chain}']), "Mutant sequence length must match wildtype sequence length"
                        # Check mutant seq to ensure mutations are all canonical amino acids
                        for wc, mc in zip(wt_info[f'seq_chain_{chain}'], mut_seq_dict[chain]):
                            if wc != mc: assert mc in mut_alphabet, "Mutation must be one of 20 canonical amino acids"
                        mut_seq.append((chain, mut_seq_dict[chain]))
                    else: # Use wildtype sequence
                        mut_seq.append((chain, wt_info[f'seq_chain_{chain}']))
                mutant_data['pdb'].append(pdb)
                mutant_data['sequences'].append(mut_seq)
                mutant_data['ddG_expt'].append(ddG_expt)
                mutant_data['mut_chains'].append(':'.join(mut_chains))
            chain_lens_dicts[pdb] = {chain: len(chain_seq) for chain, chain_seq in mutant_data['sequences'][-1]}

    elif cfg.mutant_csv is not None: # Predict energies for provided mutant sequences from a CSV file
        mutant_df = pd.read_csv(cfg.mutant_csv)
        assert all(col in mutant_df.columns for col in ['pdb', 'chain', 'mut_type']), "CSV must contain 'pdb', 'chain', and 'mut_type' columns"
        if not 'ddG_expt' in mutant_df.columns:
            mutant_df['ddG_expt'] = [np.nan] * len(mutant_df)
        for pdb in mutant_df['pdb'].unique():
            pdb_df = mutant_df[mutant_df['pdb'] == pdb]
            wt_info = parse_PDB_seq_only(os.path.join(cfg.input_dir, pdb + '.pdb'), skip_gaps=cfg.inference.skip_gaps)
            for chain_list, mut_type_list, ddG_expt in zip(pdb_df['chain'], pdb_df['mut_type'], pdb_df['ddG_expt']):
                mut_type_dict = defaultdict(list)
                for chain, mut_type in zip(chain_list.split(':'), mut_type_list.split(':')):
                    mut_type_dict[chain].append(mut_type)
                mut_seq = []
                for chain in wt_info['chain_order']:
                    mut_chain = copy.deepcopy(wt_info[f'seq_chain_{chain}'])
                    if len(mut_type_dict[chain]) > 0: # Use mutant sequence
                        for mut_type in mut_type_dict[chain]:
                            wt, pos, mut = mut_type[0], int(mut_type[1:-1]), mut_type[-1]
                            assert wt == mut_chain[pos], "Mutation information must match wildtype sequence at the mutation position"
                            assert mut in mut_alphabet, "Mutation must be one of 20 canonical amino acids"
                            mut_chain = mut_chain[:pos] + mut + mut_chain[pos+1:]
                        mut_seq.append((chain, mut_chain))
                    else: # Use wildtype sequence
                        mut_seq.append((chain, wt_info[f'seq_chain_{chain}']))
                mutant_data['pdb'].append(pdb)
                mutant_data['sequences'].append(mut_seq)
                mutant_data['ddG_expt'].append(ddG_expt)
                mutant_data['mut_chains'].append(chain_list)
            chain_lens_dicts[pdb] = {chain: len(chain_seq) for chain, chain_seq in mutant_data['sequences'][-1]}

    else: # Do a DMS screen of all single mutants
        for pdb in pdb_list:
            wt_info = parse_PDB_seq_only(os.path.join(cfg.input_dir, pdb + '.pdb'), skip_gaps=cfg.inference.skip_gaps)
            wt_chains = [(chain, wt_info[f'seq_chain_{chain}']) for chain in wt_info['chain_order']]
            for i_chain, chain in enumerate(wt_info['chain_order']):
                mut_seq = ""
                for i, wtAA in enumerate(wt_info[f'seq_chain_{chain}']):
                    if wtAA != '-':
                        for mutAA in mut_alphabet:
                            if mutAA != wtAA:
                                mut_seq = copy.deepcopy(wt_info[f'seq_chain_{chain}'])
                                mut_seq = mut_seq[:i] + mutAA + mut_seq[i+1:]
                                mutant_data['pdb'].append(pdb)
                                full_mut_seq = copy.deepcopy(wt_chains)
                                full_mut_seq[i_chain] = (chain, mut_seq)
                                mutant_data['sequences'].append(full_mut_seq)
                                mutant_data['ddG_expt'].append(np.nan)
                                mutant_data['mut_chains'].append(chain)
            chain_lens_dicts[pdb] = {chain: len(chain_seq) for chain, chain_seq in mutant_data['sequences'][-1]}

    if binding_energy_chains: # Split sequences into separate chains if requested for binding prediction
        for pdb, seq in zip(mutant_data['pdb'], mutant_data['sequences']):
            assert pdb in binding_energy_chains.keys(), "To calculate binding energies, chain partition information must be present for each structure"
            all_chains = []
            for partition in binding_energy_chains[pdb]:
                all_chains += partition
            assert sorted(all_chains) == sorted([chain for chain, _ in mutant_data['sequences'][0]]), "Chain partitions must include all chains in structure"
            partitioned_sequences = []
            for partition in binding_energy_chains[pdb]:
                partitioned_sequences.append(etab_utils.seq_to_tensor("".join([chain_seq for chain, chain_seq in seq if chain in partition]), cfg.dev))
            mutant_data['partitioned_sequences'].append(partitioned_sequences)
    else:
        mutant_data['partitioned_sequences'] = [None] * len(mutant_data['sequences'])

    # Save mutant sequences and energies to tensors
    for i_mut in range(len(mutant_data['sequences'])):
        mutant_data['sequences'][i_mut] = "".join([chain_seq for _, chain_seq in mutant_data['sequences'][i_mut]])
    
    return pdb_list, pd.DataFrame(mutant_data), chain_lens_dicts

def get_etab(model, pdb_data, cfg, partition):
    """
    Get energy table for a given PDB structure.
    
    Parameters
    ----------
    model : PottsMPNN model
        Model with which to score sequences
    pdb_data : dict
        dict with PDB information
    cfg : omegacong
        Config object
    partition : list (optional, default None)
        list of chains to analyze

    Returns
    -------
    etab : torch.Tensor
        Energy table
    E_idx : torch.Tensor
        Neighbor indices
    """
    # Featurize all chains
    partition_dict = {pdb_data[0]['name']: [partition, []]} if partition else None
    X, _, mask, _, _, chain_encoding_all, _, _, _, _, _, _, residue_idx, _, _, _, _, _, _, _, _ = tied_featurize(
        [pdb_data[0]], cfg.dev, partition_dict, None, None, 
        None, None, None, ca_only=False, vocab=cfg.model.vocab
    )

    # Run encoder
    _, E_idx, _, etab = model.run_encoder(X, mask, residue_idx, chain_encoding_all)
    etab = etab_utils.functionalize_etab(etab, E_idx)
    pad = (0, 2, 0, 2) # Pad for 'X' and '-' tokens
    etab = torch.nn.functional.pad(etab, pad, "constant", 0) # Add padding to account for 'X' and '-' tokens
    return etab, E_idx

def score_seqs(model, cfg, pdb_data, nrgs, seqs, partition=None):
    """
    Score sequences using the energy table.

    Parameters
    ----------  
    model : PottsMPNN model
        Model with which to score sequences
    cfg : omegacong
        Config object
    pdb_data : dict
        dict with PDB information
    nrgs : list of shape (N,)
        Mutant energy information
    seqs : list of shape (N, L)
        Mutant sequence information
    partition : list (optional, default None)
        list of chains to analyze
    
    Returns
    -------
    scores : torch.Tensor, shape (N,)
        Scores for each sequence.
    scored_seqs : torch.Tensor, shape (N, L)
        Scored sequences
    reference_scores : torch.Tensor, shape (N,)
        References for scored sequences
    """
    etab, E_idx = get_etab(model, pdb_data, cfg, partition)
    
    # Run energy prediction according to config
    if cfg.inference.ddG: # If ddG prediction (default), use wildtype as reference energy
        nrgs = np.insert(nrgs, 0, 0.0)
        seqs = np.insert(seqs, 0, pdb_data[0]['seq'])
    # Transform nrgs and seqs to tensors
    nrgs = torch.from_numpy(np.array(nrgs)).to(dtype=torch.float32, device=cfg.dev).unsqueeze(0)
    seqs = torch.stack([etab_utils.seq_to_tensor(seq) for seq in seqs]).to(dtype=torch.int64, device=cfg.dev).unsqueeze(0)

    if etab.size(1)*nrgs.shape[1] > cfg.inference.max_tokens:
        batch_size = int(cfg.inference.max_tokens / etab.size(1))
    else:
        batch_size = nrgs.shape[1]
    
    # Calculate energies
    scores, scored_seqs, reference_scores = [], [], []
    for batch in range(0, nrgs.shape[1], batch_size):
        batch_scores, batch_seqs, batch_refs = etab_utils.calc_eners(etab, E_idx, seqs[:,batch:batch+batch_size], nrgs[:,batch:batch+batch_size], filter=cfg.inference.filter)
        scores.append(batch_scores)
        scored_seqs.append(batch_seqs)
        reference_scores.append(batch_refs)
    scores, scored_seqs, reference_scores = torch.cat(scores, 1), torch.cat(scored_seqs, 1), torch.cat(reference_scores, 1)

    if cfg.inference.ddG: # If ddG prediction (default), compare to wildtype and remove reference
        scores = scores -scores[:, 0]
        scores, scored_seqs, reference_scores = scores[:, 1:], scored_seqs[:, 1:], reference_scores[:, 1:]

    if cfg.inference.mean_norm: # By default, normalize so mean is 0 (helps when comparing proteins with large numbers of mutants)
        scores -= torch.mean(scores, dim=1)
    return scores, scored_seqs, reference_scores

def plot_data(data,
              only_mutated_positions=False,
              title='PottsMPNN Predictions',
              clabel=r'Predicted $\Delta\Delta$G (a.u.)',
              save_path=None,
              figsize=(20, 5),
              ener_type='ddG',
              chain_ranges=None,
              chain_order=None):
    """
    Plots a heatmap of mutation energies from a dataframe.

    Parameters:
    - data : DataFrame with columns 'mutant', 'wildtype', 'ddG_pred'.
            Sequences use ':' as chain delimiters.
    - only_mutated_positions : If True, only plots columns (residues) that have at least one mutation.
    - chain_ranges : Dict { 'A': [start, stop] } defining inclusive 1-indexed ranges for specific chains.
    - chain_order : List of strings (e.g. ['H', 'L']). 
                   1. Maps the split input sequences to these names (Index 0 -> chain_order[0]).
                   2. Determines the order in which chains are plotted.
                   If None, defaults to ['A', 'B', 'C'...] and alphabetical sort.
    """

    amino_acids = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L',
                   'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y']
    aa_to_idx = {aa: i for i, aa in enumerate(amino_acids)}

    # --- 1. Parse Data ---
    parsed_data = {}     # parsed_data[chain_name][pos] = {wt, muts}
    chain_sequences = {} # chain_sequences[chain_name] = list(sequence)

    for _, row in data.iterrows():
        wt_seq = row['wildtype']
        mut_seq = row['mutant']
        energy = row['ddG_pred']

        wt_chains = wt_seq.split(':')
        mut_chains = mut_seq.split(':')

        if len(wt_chains) != len(mut_chains):
            continue 

        # Determine Chain Names for this row
        current_chain_names = []
        if chain_order:
            current_chain_names = chain_order[:len(wt_chains)]
        else:
            current_chain_names = [chr(65 + i) for i in range(len(wt_chains))]

        # Identify mutations
        global_mutations = [] 
        
        for c_name, w_chain, m_chain in zip(current_chain_names, wt_chains, mut_chains):
            if len(w_chain) != len(m_chain): continue 
            
            # Store WT sequence logic (first time we see this chain name)
            if c_name not in chain_sequences:
                chain_sequences[c_name] = list(w_chain)
            
            # Find mismatches
            for i, (w, m) in enumerate(zip(w_chain, m_chain)):
                if w != m:
                    # 1-indexed position
                    global_mutations.append((c_name, i + 1, w, m))

        # Constraint: Only single mutations allowed per row
        if len(global_mutations) == 1:
            c_name, pos, wt, mut = global_mutations[0]
            
            if c_name not in parsed_data: parsed_data[c_name] = {}
            if pos not in parsed_data[c_name]: parsed_data[c_name][pos] = {'wt': wt, 'muts': {}}
            
            parsed_data[c_name][pos]['muts'][mut] = energy

    # --- 2. Determine Chains to Plot ---
    if chain_order:
        active_chain_names = [c for c in chain_order if c in chain_sequences]
    else:
        active_chain_names = sorted(chain_sequences.keys())

    if not active_chain_names:
        print("No valid data found to plot.")
        return

    # --- 3. Construct Matrix Columns ---
    matrix_columns = []   # List of (chain_name, pos, wt_residue)
    chain_boundaries = [] # List of column indices where new chains start

    current_col_idx = 0
    for c_name in active_chain_names:
        chain_boundaries.append(current_col_idx)
        full_seq = chain_sequences[c_name]
        
        # Determine valid range for this chain
        start_r, stop_r = 1, len(full_seq)
        if chain_ranges and c_name in chain_ranges:
            start_r, stop_r = chain_ranges[c_name]
        elif chain_ranges:
            continue 

        # Determine which positions to include
        if only_mutated_positions:
            existing_pos = sorted(parsed_data.get(c_name, {}).keys())
            positions = [p for p in existing_pos if start_r <= p <= stop_r]
        else:
            actual_start = max(1, start_r)
            actual_stop = min(len(full_seq), stop_r)
            if actual_start > actual_stop:
                positions = []
            else:
                positions = range(actual_start, actual_stop + 1)

        for pos in positions:
            wt_aa = full_seq[pos - 1] # 0-indexed lookup
            matrix_columns.append((c_name, pos, wt_aa))
            current_col_idx += 1
            
    # Initialize matrix
    heatmap_data = np.full((len(amino_acids), len(matrix_columns)), np.nan)

    # Fill matrix
    for col_idx, (c_name, pos, wt_aa) in enumerate(matrix_columns):
        if ener_type == 'ddG' and wt_aa in aa_to_idx:
            heatmap_data[aa_to_idx[wt_aa], col_idx] = 0.0

        if c_name in parsed_data and pos in parsed_data[c_name]:
            muts = parsed_data[c_name][pos]['muts']
            for mut_aa, ener in muts.items():
                if mut_aa in aa_to_idx:
                    row_idx = aa_to_idx[mut_aa]
                    heatmap_data[row_idx, col_idx] = ener

    # --- 4. Plotting ---
    blue = (0.0, 0.0, 1.0)
    gray90 = (0.9, 0.9, 0.9)
    red = (1.0, 0.0, 0.0)
    cmap = mcolors.LinearSegmentedColormap.from_list("Blue_Gray90_Red", [blue, gray90, red])
    
    if ener_type == 'ddG':
        center = 0
    else:
        center = np.nanmean(heatmap_data)

    fig, ax = plt.subplots(figsize=figsize, dpi=150)
    sns.set(font_scale=0.8)
    ax.set_facecolor('#E0E0E0')

    # Prepare labels
    tick_labels = [f"{wt}{pos}" for (_, pos, wt) in matrix_columns]

    sns.heatmap(
        heatmap_data,
        cmap=cmap,
        center=center,
        yticklabels=amino_acids,
        xticklabels=False, 
        cbar_kws={'shrink': 0.8, 'pad': 0.02, 'label': clabel},
        mask=np.isnan(heatmap_data),
        ax=ax
    )
    ax.collections[0].colorbar.ax.set_ylabel(clabel, fontsize=12) 
    ax.collections[0].colorbar.ax.tick_params(labelsize=12)

    # --- 5. Styling Missing Data (Exact 'X' using Lines) ---
    segments = []
    rows, cols = heatmap_data.shape
    for r in range(rows):
        for c in range(cols):
            if np.isnan(heatmap_data[r, c]):
                p1 = (c, r)
                p2 = (c + 1, r + 1)
                p3 = (c, r + 1)
                p4 = (c + 1, r)
                segments.append([p1, p2])
                segments.append([p3, p4])
    
    if segments:
        lc = LineCollection(segments, color='gray', linewidths=0.5, alpha=0.5)
        ax.add_collection(lc)

    # --- 6. Formatting Axes & Borders ---
    for tick in ax.get_yticklabels():
        tick.set_rotation(0)
        tick.set_ha('left')
        tick.set_position((-0.02, tick.get_position()[1]))
        tick.set_fontsize(12)

    # Font size calculation
    n_cols = len(matrix_columns)
    tick_indices = np.arange(0, n_cols, 1)
    tick_locs = tick_indices + 0.5
    fig_w, _ = fig.get_size_inches()
    ax_w_frac = ax.get_position().width
    box_w_inches = (fig_w * ax_w_frac) / max(1, n_cols)
    max_font_size = box_w_inches * 72 * 0.9
    final_fontsize = min(12, max_font_size)

    ax.set_xticks(tick_locs)
    ax.set_xticklabels(tick_labels, rotation=90, fontsize=final_fontsize)

    plt.xlabel('Wildtype Residue', fontsize=12, labelpad=25) 
    plt.ylabel('Mutant Residue', fontsize=12)
    plt.title(title, fontsize=12)

    # Add Borders around Chains & Chain Labels
    boundaries = chain_boundaries + [len(matrix_columns)]
    
    for i, c_name in enumerate(active_chain_names):
        if chain_ranges and c_name not in chain_ranges:
            continue
        start = boundaries[i]
        end = boundaries[i+1]
        width = end - start
        height = len(amino_acids)
        
        # 1. Draw Border
        rect = Rectangle((start, 0), width, height, 
                         fill=False, edgecolor='black', lw=2, clip_on=False)
        ax.add_patch(rect)

        # 2. Add Chain Label
        # Calculate center in data coordinates (x-axis)
        center_x = (start + end) / 2
        
        ax.text(center_x, -0.14, f"Chain {c_name}", 
                ha='center', va='top', fontsize=12, fontweight='bold',
                transform=ax.get_xaxis_transform())

    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def rewrite_pdb_sequences(pdb_dict, pdb_in_dir, pdb_out_dir):
    """
    Save new .pdb files with updated sequences
    
    Parameters
    ----------
    pdb_dict : dict
        Keys: "<pdb>|<vis_chains>|<hidden_chains>"
        Values: (seq_string, sample)
            seq_string: ':'-separated sequences for vis chains then hidden chains
            sample: int (unused, but accepted)
    pdb_in_dir : str
        Input directory
    pdb_out_dir : str
        Output directory
    """
    # Reverse map: 1-letter -> 3-letter
    AA1_TO_AA3 = {k.upper(): v.upper() for k, v in protein_letters_1to3.items()}

    os.makedirs(pdb_out_dir, exist_ok=True)
    parser = PDBParser(QUIET=True)
    io = PDBIO()

    for key, (seq_string, _) in pdb_dict.items():
        # Get chain info
        chain_info = key.split("|")
        if len(chain_info) == 3:
            pdb_name, hidden_chains_str, vis_chains_str = chain_info
            vis_chains = vis_chains_str.split(":") if vis_chains_str else []
            hidden_chains = hidden_chains_str.split(":") if hidden_chains_str else []
            all_chains = hidden_chains + vis_chains
            if len(vis_chains) > 0:
                out_name = f"{pdb_name}_{hidden_chains_str.replace(':', '-')}_{vis_chains_str.replace(':', '-')}.pdb"
            else:
                out_name = f"{pdb_name}_{hidden_chains_str.replace(':', '-')}.pdb"
        else:
            pdb_name = chain_info[0]
            wt_info = parse_PDB_seq_only(os.path.join(pdb_in_dir, pdb_name + '.pdb'))
            vis_chains = []
            all_chains = wt_info['chain_order']
            out_name = f"{pdb_name}.pdb"

        seqs = seq_string.split(":")
        if len(seqs) != len(all_chains):
            raise ValueError(
                f"Sequence count ({len(seqs)}) does not match chain count "
                f"({len(all_chains)}) for {key}"
            )

        structure = parser.get_structure(
            pdb_name, os.path.join(pdb_in_dir, pdb_name + ".pdb")
        )

        chain_to_seq = dict(zip(all_chains, seqs))

        for model in structure:
            for chain in model:
                chain_id = chain.id
                if chain_id not in chain_to_seq:
                    continue

                raw_seq = chain_to_seq[chain_id]
                is_visible = chain_id in vis_chains

                residues = [
                    res for res in chain
                    if res.id[0] == " "
                ]

                if is_visible:
                    # ----------------------------
                    # Visible chain: gap-aware
                    # ----------------------------
                    seq_i = 0

                    for res in residues:
                        if seq_i >= len(raw_seq):
                            raise ValueError(
                                f"Sequence too short for visible chain {chain_id} in {key}"
                            )

                        aa1 = raw_seq[seq_i].upper()
                        seq_i += 1

                        # Gap → keep original residue
                        if aa1 == "-":
                            continue

                        if aa1 not in AA1_TO_AA3:
                            raise ValueError(
                                f"Invalid amino acid '{aa1}' in chain {chain_id} in {key}"
                            )

                        res.resname = AA1_TO_AA3[aa1]

                    if seq_i != len(raw_seq):
                        raise ValueError(
                            f"Sequence not fully consumed for visible chain {chain_id} in {key}"
                        )

                else:
                    # ----------------------------
                    # Hidden chain: strip gaps
                    # ----------------------------
                    seq = raw_seq.replace("-", "")
                    if len(seq) != len(residues):
                        raise ValueError(
                            f"Length mismatch for hidden chain {chain_id} in {key}: "
                            f"{len(residues)} residues vs {len(seq)} sequence"
                        )

                    for res, aa1 in zip(residues, seq):
                        aa1 = aa1.upper()
                        if aa1 not in AA1_TO_AA3:
                            raise ValueError(
                                f"Invalid amino acid '{aa1}' in chain {chain_id} in {key}"
                            )
                        res.resname = AA1_TO_AA3[aa1]
                        
        out_path = os.path.join(pdb_out_dir, out_name)

        io.set_structure(structure)
        io.save(out_path)

def chain_to_partition_map(
    chain_encoding_all: torch.Tensor,
    chains: list[str],
    partitions: list[list[str]],
) -> torch.Tensor:
    """
    Convert chain indices to partition indices.

    Parameters
    ----------
    chain_encoding_all : torch.Tensor
        Shape (1, L), 1-indexed chain indices
    chains : list[str]
        Ordered list of chain IDs defining the chain index mapping
    partitions : list[list[str]]
        List of chain partitions

    Returns
    -------
    partition_encoding_all : torch.Tensor
        Shape (1, L), 0-indexed partition indices
    """

    if chain_encoding_all.ndim != 2 or chain_encoding_all.shape[0] != 1:
        raise ValueError(
            "chain_encoding_all must have shape (1, L)"
        )

    # -----------------------------
    # Validation
    # -----------------------------
    chain_set = set(chains)

    part_chain_list = [c for part in partitions for c in part]
    part_chain_set = set(part_chain_list)

    unknown = part_chain_set - chain_set
    if unknown:
        raise ValueError(f"Partitions contain unknown chains: {unknown}")

    missing = chain_set - part_chain_set
    if missing:
        raise ValueError(f"Chains not covered by partitions: {missing}")

    if len(part_chain_list) != len(part_chain_set):
        raise ValueError("A chain appears in more than one partition")

    # -----------------------------
    # Build chain_idx → partition_idx map
    # -----------------------------
    # chain index is 1-indexed
    # partition index is 0-indexed
    chain_to_partition = {}

    for p_idx, part in enumerate(partitions):
        for c in part:
            chain_to_partition[c] = p_idx

    idx_map = torch.empty(len(chains) + 1, dtype=torch.long)

    for i, chain in enumerate(chains, start=1):
        idx_map[i] = chain_to_partition[chain]

    # -----------------------------
    # Remap tensor
    # -----------------------------
    if chain_encoding_all.min() < 1 or chain_encoding_all.max() > len(chains):
        raise ValueError("chain_encoding_all contains invalid chain indices")

    partition_encoding_all = idx_map[chain_encoding_all]

    return partition_encoding_all


### Currently unused functions:
def best_mutant_per_position(scores: torch.Tensor, sequences: torch.Tensor, threshold: float = 0.0) -> torch.Tensor:
    """
    Find the best scoring residue at each position with a score threshold.
    Lower scores are better. Wildtype is at index 0 and always has score 0.
    
    Parameters
    ----------
    scores : torch.Tensor, shape (N,)
        Scores for each sequence.
    sequences : torch.Tensor, shape (N, L)
        Encoded sequences (e.g., ints for residues).
        N = L*19 + 1: one wildtype + all single mutants.
    threshold : float
        Mutant must beat this score (≤ threshold) to replace WT.
    
    Returns
    -------
    torch.Tensor, shape (L,)
        Sequence of best residues at each position.
    """
    N, L = sequences.shape
    assert scores.shape[0] == N
    assert N == L * 19 + 1, "N must equal L*19 + 1"

    # Wildtype is the first entry
    native_seq = sequences[0].clone()

    # Mutants: skip the first entry
    mutant_scores = scores[1:].reshape(L, 19)      # (L, 19)
    mutant_seqs = sequences[1:].reshape(L, 19, L)  # (L, 19, L)

    # Best mutant score and residue for each position
    best_scores, best_idx = mutant_scores.min(dim=1)  # low = good
    row_idx = torch.arange(L, device=scores.device)
    best_residues = mutant_seqs[row_idx, best_idx, row_idx]

    # Apply threshold: only replace WT if mutant is good enough
    use_mutant = best_scores <= threshold
    best_seq = native_seq.clone()
    best_seq[use_mutant] = best_residues[use_mutant]

    return best_seq

def gather_nodes_t(nodes, neighbor_idx):
    """
    Gather neighbor node features for a single neighbor index array.

    Parameters
    ----------
    nodes : torch.Tensor, shape (B, N, C)
    neighbor_idx : torch.Tensor, shape (B, K)
        Single list of neighbor indices per batch.

    Returns
    -------
    torch.Tensor, shape (B, K, C)
        Neighbor features for each provided index.
    """
    # Expand indices to include channel dimension then gather
    idx_flat = neighbor_idx.unsqueeze(-1).expand(-1, -1, nodes.size(2))
    neighbor_features = torch.gather(nodes, 1, idx_flat)
    return neighbor_features

def get_ssm_mutation(pdb, seq_pos):
    """
    Build a list of single-site mutations (SSM) for a given sequence position.

    Parameters
    ----------
    pdb : dict-like
        Parsed PDB entry containing a `'seq'` string of residues (gaps as '-')
    seq_pos : int
        Position in the sequence to create mutations for.

    Returns
    -------
    list[str | None]
        List of mutation strings like 'A12G' for all canonical amino acids.
        If the residue is missing ('-') returns a list with a single `None`.
    """
    ALPHABET = 'ACDEFGHIKLMNPQRSTVWYX'
    mutation_list = []
    wtAA = pdb['seq'][seq_pos]
    # If residue exists, add all single-residue substitutions (excluding X)
    if wtAA != '-':
        for mutAA in ALPHABET[:-1]:
            mutation_list.append(wtAA + str(seq_pos) + mutAA)
    else:
        mutation_list.append(None)

    return mutation_list

def get_ssm_mutations(pdb):
    """
    Build a flat list of single-site mutations for all positions in a PDB entry.

    Parameters
    ----------
    pdb : dict-like
        Parsed PDB entry containing a `'seq'` string of residues (gaps as '-')

    Returns
    -------
    list[str | None]
        Flattened list of mutation strings for every sequence position. If a
        position is missing (gap) the list contains `None` for that position.
    """
    ALPHABET = 'ACDEFGHIKLMNPQRSTVWYX'
    mutation_list = []
    for seq_pos in range(len(pdb['seq'])):
        wtAA = pdb['seq'][seq_pos]
        if wtAA != '-':
            for mutAA in ALPHABET[:-1]:
                mutation_list.append(wtAA + str(seq_pos) + mutAA)
        else:
            mutation_list.append(None)

    return mutation_list