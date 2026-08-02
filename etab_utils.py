"""
Code in this file was originally written by or adapted from:

    1) Li et al. "Neural network-derived Potts models for structure-based protein
    design using backbone atomic coordinates and tertiary motifs"
    Protein Science, 2023. doi:10.1002/pro.4554
        - Code: https://github.com/KeatingLab/terminator

    2) Birnbaum and Keating "Beyond native sequence recovery: Improved modeling of the
    sequence-energy landscape of protein structures"
    bioRxiv, 2026. doi:10.64898/2026.01.14.699067
       - Code: https://github.com/KeatingLab/PottsMPNN
"""

import sys
import torch
import numpy as np

def merge_duplicate_pairE(h_E, E_idx, denom=2):
    """ Average pair energy tables across bidirectional edges.

    TERMinator edges are represented as two bidirectional edges, and to allow for
    communication between these edges we average the embeddings. In the case for
    pair energies, we transpose the tables to ensure that the pair energy table
    is symmetric upon inverse (e.g. the pair energy between i and j should be
    the same as the pair energy between j and i)

    Args
    ----
    h_E : torch.Tensor
        Pair energies in kNN sparse form
        Shape : n_batch x n_res x k x n_aa x n_aa
    E_idx : torch.LongTensor
        kNN sparse edge indices
        Shape : n_batch x n_res x k

    Returns
    -------
    torch.Tensor
        Pair energies with merged energies for bidirectional edges
        Shape : n_batch x n_res x k x n_aa x n_aa
    """
    try:
        k = E_idx.shape[-1]
        seq_lens = torch.ones(h_E.shape[0]).long().to(h_E.device) * h_E.shape[1]
        h_E_geometric = h_E.view([-1, 400])
        split_E_idxs = torch.unbind(E_idx)
        offset = [seq_lens[:i].sum() for i in range(len(seq_lens))]
        split_E_idxs = [e.to(h_E.device) + o for e, o in zip(split_E_idxs, offset)]
        edge_index_row = torch.cat([e.view(-1) for e in split_E_idxs], dim=0)
        edge_index_col = torch.repeat_interleave(torch.arange(edge_index_row.shape[0] // k), k).to(h_E.device)
        edge_index = torch.stack([edge_index_row, edge_index_col])
        merge = merge_duplicate_pairE_geometric(h_E_geometric, edge_index, k, denom=denom)
        merge = merge.view(h_E.shape)

        return merge
    except RuntimeError as err:
        print(err, file=sys.stderr)
        print("We're handling this error as if it's an out-of-memory error", file=sys.stderr)
        torch.cuda.empty_cache()  # this is probably unnecessary but just in case
        return merge_duplicate_pairE_sparse(h_E, E_idx)


def merge_duplicate_pairE_sparse(h_E, E_idx):
    """ Sparse method to average pair energy tables across bidirectional edges.

    Note: This method involves a significant slowdown so it's only worth using if memory is an issue.

    TERMinator edges are represented as two bidirectional edges, and to allow for
    communication between these edges we average the embeddings. In the case for
    pair energies, we transpose the tables to ensure that the pair energy table
    is symmetric upon inverse (e.g. the pair energy between i and j should be
    the same as the pair energy between j and i)

    Args
    ----
    h_E : torch.Tensor
        Pair energies in kNN sparse form
        Shape : n_batch x n_res x k x n_aa x n_aa
    E_idx : torch.LongTensor
        kNN sparse edge indices
        Shape : n_batch x n_res x k

    Returns
    -------
    torch.Tensor
        Pair energies with merged energies for bidirectional edges
        Shape : n_batch x n_res x k x n_aa x n_aa
    """
    dev = h_E.device
    n_batch, n_nodes, k, n_aa, _ = h_E.shape
    # convert etab into a sparse etab
    # self idx of the edge
    ref_idx = E_idx[:, :, 0:1].expand(-1, -1, k)
    # sparse idx
    g_idx = torch.cat([E_idx.unsqueeze(1), ref_idx.unsqueeze(1)], dim=1)
    sparse_idx = g_idx.view([n_batch, 2, -1])
    # generate a 1D idx for the forward and backward direction
    scaler = torch.ones_like(sparse_idx).to(dev)
    scaler = scaler * n_nodes
    scaler_f = scaler
    scaler_f[:, 0] = 1
    scaler_r = torch.flip(scaler_f, [1])
    batch_offset = torch.arange(n_batch).unsqueeze(-1).expand([-1, n_nodes * k]) * n_nodes * k
    batch_offset = batch_offset.to(dev)
    sparse_idx_f = torch.sum(scaler_f * sparse_idx, 1) + batch_offset
    flat_idx_f = sparse_idx_f.view([-1])
    sparse_idx_r = torch.sum(scaler_r * sparse_idx, 1) + batch_offset
    flat_idx_r = sparse_idx_r.view([-1])
    # generate sparse tensors
    flat_h_E_f = h_E.view([n_batch * n_nodes * k, n_aa**2])
    reverse_h_E = h_E.transpose(-2, -1).contiguous()
    flat_h_E_r = reverse_h_E.view([n_batch * n_nodes * k, n_aa**2])
    sparse_etab_f = torch.sparse_coo_tensor(flat_idx_f.unsqueeze(0), flat_h_E_f,
                                            (n_batch * n_nodes * n_nodes, n_aa**2))
    count_f = torch.sparse_coo_tensor(flat_idx_f.unsqueeze(0), torch.ones_like(flat_idx_f),
                                      (n_batch * n_nodes * n_nodes, ))
    sparse_etab_r = torch.sparse_coo_tensor(flat_idx_r.unsqueeze(0), flat_h_E_r,
                                            (n_batch * n_nodes * n_nodes, n_aa**2))
    count_r = torch.sparse_coo_tensor(flat_idx_r.unsqueeze(0), torch.ones_like(flat_idx_r),
                                      (n_batch * n_nodes * n_nodes, ))
    # merge
    sparse_etab = sparse_etab_f + sparse_etab_r
    sparse_etab = sparse_etab.coalesce()
    count = count_f + count_r
    count = count.coalesce()

    # this step is very slow, but implementing something faster is probably a lot of work
    # requires pytorch 1.10 to be fast enough to be usable
    collect = sparse_etab.index_select(0, flat_idx_f).to_dense()
    weight = count.index_select(0, flat_idx_f).to_dense()

    flat_merged_etab = collect / weight.unsqueeze(-1)
    merged_etab = flat_merged_etab.view(h_E.shape)
    return merged_etab


def merge_duplicate_pairE_geometric(h_E, edge_index, k, denom=2):
    """ Sparse method to average pair energy tables across bidirectional edges with Torch Geometric.

    TERMinator edges are represented as two bidirectional edges, and to allow for
    communication between these edges we average the embeddings. In the case for
    pair energies, we transpose the tables to ensure that the pair energy table
    is symmetric upon inverse (e.g. the pair energy between i and j should be
    the same as the pair energy between j and i)

    This function assumes edge_index is sorted by columns, and will fail if
    this is not the case.

    Args
    ----
    h_E : torch.Tensor
        Pair energies in Torch Geometric sparse form
        Shape : n_edge x 400
    E_idx : torch.LongTensor
        Torch Geometric sparse edge indices
        Shape : 2 x n_edge
    k : int
        Number of neighbors (including self) for each node

    Returns
    -------
    torch.Tensor
        Pair energies with merged energies for bidirectional edges
        Shape : n_edge x 400
    """
    num_nodes = edge_index.max() + 1
    row_idx = edge_index[0] + edge_index[1] * num_nodes
    col_idx = edge_index[1] + edge_index[0] * num_nodes
    internal_idx = torch.arange(edge_index.shape[1]).to(h_E.device)

    mapping = torch.zeros(max(row_idx.max(), col_idx.max()) + 1).long().to(h_E.device) - 1
    mapping[col_idx] = internal_idx

    reverse_idx = mapping[row_idx]
    mask = (reverse_idx >= 0)
    if denom != 2:
        mask[(reverse_idx % k == 0)] = False
    reverse_idx = reverse_idx[mask]

    reverse_h_E = h_E[mask]
    transpose_h_E = reverse_h_E.view([-1, 20, 20]).transpose(-1, -2).reshape([-1, 400])
    h_E[reverse_idx] = (h_E[reverse_idx] + transpose_h_E)/denom

    return h_E

def expand_etab(etab, idxs):
    """
    Expand energy table to sparse representation.

    Args
    ----
    etab : torch.Tensor
        Dense energy table of shape [B, L, k, h, h]
    idxs : torch.LongTensor
        Indices to expand along dimension 2 of shape [B, L, k]
    
    Returns
    -------
    cetab : torch.Tensor
        Expanded energy table of shape [B, L, L, h, h]
    """
    h = etab.shape[-1]
    tetab = etab.to(dtype=torch.float64)
    eidx = idxs.unsqueeze(-1).unsqueeze(-1).expand(etab.shape)
    netab = torch.zeros(tetab.shape[0], tetab.shape[1], tetab.shape[1], h, h, dtype=torch.float64, device=etab.device)
    netab.scatter_(2, eidx, tetab)
    cetab = netab.transpose(1,2).transpose(3,4)
    cetab.scatter_(2, eidx, tetab)
    return cetab

def positional_potts_energy(etab, E_idx, seq, pos):
    """
    Quick per-position Potts energy calculation using an energy table.

    This function computes the energy contribution for every possible amino
    acid at `pos` given the energy table `etab`, the neighbor indices `E_idx`
    and the current sequence `seq`. It returns a vector of length 20 with the
    energy for each amino acid at that position.

    Args
    ----------
    etab : torch.Tensor
        Energy table with shape `[B, L, k, h, h]` where the last two dims are
        square matrices (e.g., 20 x 20). Often B=1 for single-protein use.
    E_idx : torch.Tensor
        Neighbor indices used to lookup pair contributions.
    seq : torch.Tensor
        Encoded sequence tensor `[B, L]` of integer amino-acid codes.
    pos : int
        Position index to compute energies for.

    Returns
    -------
    torch.Tensor, shape (h,)
        Energy for each amino acid at the requested position.
    """
    b, L, k, h, _ = etab.shape
    # Self interaction (diagonal) for focal position
    self_etab = etab[0, pos, 0:1]  # 1 x h x h
    pair_etab = etab[0, pos, 1:]   # (k-1) x h x h
    # Extract diagonal (self) energies -> length h
    self_nrgs_im = torch.diagonal(self_etab, offset=0, dim1=-2, dim2=-1).squeeze()
    # Gather neighbor indices and the amino acids at neighbor positions
    E_idx_jn = E_idx[0, pos, 1:].unsqueeze(-1)  # (k-1) x 1
    E_aa = torch.gather(seq[0].unsqueeze(-1).expand(-1, k - 1), 0, E_idx_jn)
    # Expand to match pair_etab last-dimension for gather
    E_aa = E_aa.view(list(E_idx_jn.shape) + [1]).expand(-1, h, -1)
    # Gather pair energies for each candidate AA and sum across neighbors
    pair_nrgs_jn = torch.gather(pair_etab, -1, E_aa).squeeze(-1).sum(0)
    return self_nrgs_im + pair_nrgs_jn

def functionalize_etab(etab, E_idx):
    """
    Prepare etab for energy prediction.

    Args
    ----
    etab : torch.Tensor
        Energy table with shape `[B, L, k, h]`
    E_idx : torch.Tensor
        Neighbor indices used to lookup pair contributions.

    Returns
    -------
    torch.Tensor, shape (B, L, k, h, h)
        Energy table ready for energy prediction.
    """
    etab = merge_duplicate_pairE(etab, E_idx, denom=4)
    etab = etab.view(etab.shape[0], etab.shape[1], etab.shape[2], int(np.sqrt(etab.shape[3])), int(np.sqrt(etab.shape[3])))
    return etab

def calc_eners(etab, E_idx, seqs, nrgs, filter=True):
    """
    Calculate Potts energies for a batch of sequences using a dense energy table.

    Args
    ----
    etab : torch.Tensor
        Energy table of shape [B, L, k, h, h]
    E_idx : torch.LongTensor
        Neighbor indices of shape [B, L, k]
    seqs : torch.LongTensor
        Encoded sequences of shape [B, n, L]
    nrgs : torch.Tensor or None
        Reference energies of shape [B, n]
    filter : bool, optional 
        Whether to filter out sequences with NaN reference energies, by default True
    
    Returns
    -------
    batch_scores : torch.Tensor
        Calculated energies of shape [B, filt_n]
    seqs : torch.LongTensor
        Filtered sequences of shape [filt_n , L]
    nrgs : torch.Tensor
        Filtered reference energies of shape [filt_n]
    """
    b, n, l = seqs.shape
    h = etab.shape[-1]
    k = E_idx.shape[-1]
    etab = etab.unsqueeze(1).expand(b, n, l, k, h, h)
    E_idx = E_idx.unsqueeze(1).expand(b, n, l, k)
    E_aa_jn = torch.gather(seqs.unsqueeze(-1).expand(-1, -1, -1, k), 2, E_idx)
    E_aa_jn = E_aa_jn.view([b, n, l, k, 1, 1]).expand(-1, -1, -1, -1, h, -1)
    E_aa_im = seqs.unsqueeze(-1).expand(-1, -1, -1, k).unsqueeze(-1)
    nrgs_jn = torch.gather(etab, 5, E_aa_jn).squeeze(-1) # b x n x L x k x h
    energies = torch.gather(nrgs_jn, 4, E_aa_im).squeeze(-1) # b x n x L x k
    batch_scores = energies.sum(dim=(2, 3))

    if filter and nrgs is not None:
        mask = nrgs != torch.nan
        seqs = seqs[mask]
        nrgs = nrgs[mask]
        batch_pred_E = batch_scores[mask]
        return batch_pred_E, seqs, nrgs
    else:
        return batch_scores, seqs, nrgs

# zero is used as padding
AA_to_int = {
    'A': 1,
    'ALA': 1,
    'C': 2,
    'CYS': 2,
    'D': 3,
    'ASP': 3,
    'E': 4,
    'GLU': 4,
    'F': 5,
    'PHE': 5,
    'G': 6,
    'GLY': 6,
    'H': 7,
    'HIS': 7,
    'I': 8,
    'ILE': 8,
    'K': 9,
    'LYS': 9,
    'L': 10,
    'LEU': 10,
    'M': 11,
    'MET': 11,
    'N': 12,
    'ASN': 12,
    'P': 13,
    'PRO': 13,
    'Q': 14,
    'GLN': 14,
    'R': 15,
    'ARG': 15,
    'S': 16,
    'SER': 16,
    'T': 17,
    'THR': 17,
    'V': 18,
    'VAL': 18,
    'W': 19,
    'TRP': 19,
    'Y': 20,
    'TYR': 20,
    '-': 21,
    'X': 22
}

AA_to_int = {key: val - 1 for key, val in AA_to_int.items()}

int_to_AA = {y: x for x, y in AA_to_int.items() if len(x) == 1}

int_to_3lt_AA = {y: x for x, y in AA_to_int.items() if len(x) == 3}

def seq_to_ints(sequence):
    """
    Given a string of one-letter encoded AAs, return its corresponding integer encoding
    """
    return [AA_to_int[residue] for residue in sequence]

def seq_to_tensor(sequence, dev='cpu'):
    """
    Given a string of one-letter encoded AAs, return its corresponding integer encoding as a PyTorch tensor
    """
    return torch.tensor(seq_to_ints(sequence), dtype=torch.int64).to(device=dev)

_SEQ_LUT = None

def _get_seq_lut():
    """Lazily build a 256-entry ASCII -> integer lookup table from AA_to_int."""
    global _SEQ_LUT
    if _SEQ_LUT is None:
        lut = np.full(256, -1, dtype=np.int64)
        for aa, idx in AA_to_int.items():
            if len(aa) == 1:
                lut[ord(aa)] = idx
        _SEQ_LUT = lut
    return _SEQ_LUT

def seqs_to_tensor(sequences, dev='cpu'):
    """
    Vectorized batch version of seq_to_tensor for equal-length sequences.

    Given an iterable of equal-length one-letter sequences, return an int64
    tensor of shape (n, L). Equivalent to
    ``torch.stack([seq_to_tensor(s) for s in sequences])`` but encodes with a
    single vectorized numpy lookup instead of a per-residue Python dict loop.
    """
    sequences = list(sequences)
    if not sequences:
        return torch.empty((0, 0), dtype=torch.int64, device=dev)
    lut = _get_seq_lut()
    n = len(sequences)
    buf = np.frombuffer("".join(sequences).encode("ascii"), dtype=np.uint8)
    ints = lut[buf]
    if (ints < 0).any():
        raise ValueError("Encountered residue not present in AA_to_int mapping.")
    return torch.from_numpy(ints.reshape(n, -1)).to(device=dev)

def ints_to_seq(int_list):
    return [int_to_AA[i] if i in int_to_AA.keys() else 'X' for i in int_list]

def aa_three_to_one(residue):
    return int_to_AA[AA_to_int[residue]]

def ints_to_seq_torch(seq):
    return "".join(ints_to_seq(seq.cpu().numpy()))

def ints_to_seq_normal(seq):
    return "".join(ints_to_seq(seq))