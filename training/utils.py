import torch
from torch.utils.data import DataLoader
import csv
from dateutil import parser
import numpy as np
import time
import random
import pickle
import json
import os
from collections import defaultdict
import re
from typing import Optional
import dataclasses
import edlib
from Bio.PDB import PDBParser, is_aa
from Bio.SeqUtils import seq1
from model_utils import featurize
# Define the 14 standard atom names (N, CA, C, O, CB + most common sidechain atoms)
ATOM_ORDER = ['N', 'CA', 'C', 'O', 'CB', 'CG', 'CG1', 'CG2',
              'CD', 'CD1', 'CD2', 'CE', 'CE1', 'CE2']

STANDARD_AA_RESNAMES = {
    'ALA', 'ARG', 'ASN', 'ASP', 'CYS', 'GLN', 'GLU', 'GLY',
    'HIS', 'ILE', 'LEU', 'LYS', 'MET', 'PHE', 'PRO', 'SER',
    'THR', 'TRP', 'TYR', 'VAL'
}

class StructureDataset():
    def __init__(self, pdb_dict_list, verbose=True, truncate=None, max_length=100,
        alphabet='ACDEFGHIKLMNPQRSTVWYX'):
        alphabet_set = set([a for a in alphabet])
        discard_count = {
            'bad_chars': 0,
            'too_long': 0,
            'bad_seq_length': 0
        }

        self.data = []

        for i, entry in enumerate(pdb_dict_list):
            seq = entry['seq']

            bad_chars = set([s for s in seq]).difference(alphabet_set)
            if len(bad_chars) == 0:
                if len(entry['seq']) <= max_length:
                    self.data.append(entry)
                else:
                    discard_count['too_long'] += 1
            else:
                discard_count['bad_chars'] += 1

            # Truncate early
            if truncate is not None and len(self.data) == truncate:
                return
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if isinstance(idx, list):
            return [self.data[i] for i in idx]
        return self.data[idx]

class StructureSampler(Sampler):
    def __init__(self, dataset, batch_size=100, device='cpu', flex_type="", augment_eps=0, replicate=1, esm=None, batch_converter=None, esm_embed_layer=36, esm_embed_dim=2560, one_hot=False, openfold_backbone=False, msa_seqs=False, msa_batch_size=1):
        self.size = len(dataset)
        self.lengths = [len(dataset[i]['seq']) for i in range(self.size)]
        self.dataset = dataset
        self.batch_size = batch_size
        self.device = device
        self.flex_type = flex_type
        self.augment_eps = augment_eps
        self.replicate = replicate
        self.epoch = -1
        self.esm = esm
        self.batch_converter = batch_converter
        self.esm_embed_layer = esm_embed_layer
        self.esm_embed_dim = esm_embed_dim
        self.one_hot = one_hot
        self.openfold_backbone = openfold_backbone
        self.msa_seqs = msa_seqs
        self.msa_batch_size = msa_batch_size
        self._cluster()

    def _set_epoch(self, epoch):
        self.epoch = epoch

    def _cluster(self):
        sorted_ix = np.argsort(self.lengths)

        # Cluster into batches of similar sizes
        clusters, batch = [], []
        batch_max = 0
        for ix in sorted_ix:
            if self.batch_size == 1:
                batch.append(ix)
                clusters.append(batch)
                batch, batch_max = [], 0
                continue
            size = self.lengths[ix]
            if size * (len(batch) + 1) <= self.batch_size:
                batch.append(ix)
                batch_max = size
            else:
                clusters.append(batch)
                batch, batch_max = [], 0
        if len(batch) > 0:
            clusters.append(batch)
        self.clusters = clusters

    def package(self, b_idx):
        return featurize(b_idx, self.epoch, msa_seqs=self.msa_seqs, msa_batch_size=self.msa_batch_size)

    def __len__(self):
        return len(self.clusters)

    def __iter__(self):
        np.random.shuffle(self.clusters)
        for b_idx in self.clusters:
            yield b_idx


def worker_init_fn(worker_id):
    np.random.seed()

class NoamOpt:
    "Optim wrapper that implements rate."
    def __init__(self, model_size, factor, warmup, optimizer, step):
        self.optimizer = optimizer
        self._step = step
        self.warmup = warmup
        self.factor = factor
        self.model_size = model_size
        self._rate = 0

    @property
    def param_groups(self):
        """Return param_groups."""
        return self.optimizer.param_groups

    def step(self):
        "Update parameters and rate"
        self._step += 1
        rate = self.rate()
        for p in self.optimizer.param_groups:
            p['lr'] = rate
        self._rate = rate
        self.optimizer.step()

    def rate(self, step = None):
        "Implement `lrate` above"
        if step is None:
            step = self._step
        return self.factor * \
            (self.model_size ** (-0.5) *
            min(step ** (-0.5), step * self.warmup ** (-1.5)))

    def zero_grad(self):
        self.optimizer.zero_grad()

def get_std_opt(parameters, d_model, step):
    return NoamOpt(
        d_model, 2, 4000, torch.optim.Adam(parameters, lr=0, betas=(0.9, 0.98), eps=1e-9), step
    )

# Sequences coming from UniProtKB database come in the
# `db|UniqueIdentifier|EntryName` format, e.g. `tr|A0A146SKV9|A0A146SKV9_FUNHE`
# or `sp|P0C2L1|A3X1_LOXLA` (for TREMBL/Swiss-Prot respectively).
_UNIPROT_PATTERN = re.compile(
    r"""
    ^
    # UniProtKB/TrEMBL or UniProtKB/Swiss-Prot
    (?:tr|sp)
    \|
    # A primary accession number of the UniProtKB entry.
    (?P<AccessionIdentifier>[A-Za-z0-9]{6,10})
    # Occasionally there is a _0 or _1 isoform suffix, which we ignore.
    (?:_\d)?
    \|
    # TREMBL repeats the accession ID here. Swiss-Prot has a mnemonic
    # protein ID code.
    (?:[A-Za-z0-9]+)
    _
    # A mnemonic species identification code.
    (?P<SpeciesIdentifier>([A-Za-z0-9]){1,5})
    # Small BFD uses a final value after an underscore, which we ignore.
    (?:_\d+)?
    $
    """,
    re.VERBOSE)


@dataclasses.dataclass(frozen=True)
class Identifiers:
  species_id: str = ''


def _parse_sequence_identifier(msa_sequence_identifier: str) -> Identifiers:
  """Gets species from an msa sequence identifier.

  The sequence identifier has the format specified by
  _UNIPROT_TREMBL_ENTRY_NAME_PATTERN or _UNIPROT_SWISSPROT_ENTRY_NAME_PATTERN.
  An example of a sequence identifier: `tr|A0A146SKV9|A0A146SKV9_FUNHE`

  Args:
    msa_sequence_identifier: a sequence identifier.

  Returns:
    An `Identifiers` instance with species_id. These
    can be empty in the case where no identifier was found.
  """
  matches = re.search(_UNIPROT_PATTERN, msa_sequence_identifier.strip())
  if matches:
    return Identifiers(
        species_id=matches.group('SpeciesIdentifier'))
  return Identifiers()


def _extract_sequence_identifier(description: str) -> Optional[str]:
  """Extracts sequence identifier from description. Returns None if no match."""
  split_description = description.split()
  if split_description:
    return split_description[0].partition('/')[0]
  else:
    return None

def get_identifiers(description: str) -> Identifiers:
  """Computes extra MSA features from the description."""
  sequence_identifier = _extract_sequence_identifier(description)
  if sequence_identifier is None:
    return Identifiers()
  else:
    return _parse_sequence_identifier(sequence_identifier)

def clean_a3m(seq):
    """Remove insertions (lowercase letters)"""
    return re.sub(r'[a-z]', '', seq)

def parse_a3m_stats(filepath, n=None, species=False, source=None, dataset='ingr'):
    sequences = []
    headers = []

    with open(filepath, 'r') as f:
        current_seq = []
        for line in f:
            line = line.strip()
            if line.startswith('>') or line.startswith('#'):
                headers.append(line[1:])
                if current_seq:
                    sequences.append(''.join(current_seq))
                    current_seq = []
            else:
                current_seq.append(line)
        if current_seq:
            sequences.append(''.join(current_seq))  # Add last sequence

    # Reference (first) sequence
    ref_seq_raw = sequences[0]
    ref_seq = clean_a3m(ref_seq_raw)

    result = defaultdict(dict)
    
    for i, (seq_raw, header) in enumerate(zip(sequences, headers)):
        if (i == 0) and (dataset == 'ingr'):
            header = 'query'
        if 'B' in seq_raw or 'J' in seq_raw or 'O' in seq_raw or 'U' in seq_raw or 'Z' in seq_raw:
            continue
        insert_pct = len(re.findall(r"[a-z]", seq_raw)) / len(seq_raw)
        raw_len = len(seq_raw)
        seq = clean_a3m(seq_raw)
        aligned_len = min(len(ref_seq), len(seq))
        
        identical = sum(ref_seq[j] == seq[j] for j in range(aligned_len))
        deletions = seq.count('-')
        mutations = sum(ref_seq[j] != seq[j] for j in range(aligned_len))

        identity = identical / aligned_len if aligned_len > 0 else 0.0
        deletion_pct = deletions / len(seq) if len(seq) > 0 else 0.0
        mutation_pct = mutations / aligned_len if aligned_len > 0 else 0.0
        if source is None or header=='query':
            try:
                header = header.split(' ')[0].strip('>')
            except:
                pass
            result[header][identity] = (seq, identity, deletion_pct, mutation_pct, insert_pct, raw_len)
        elif source == 'uniref':
            try:
                species = header.split(' ')[-2].split('=')[-1]
            except:
                species = header
            result[species][identity] = (seq, identity, deletion_pct, mutation_pct, insert_pct, raw_len)
        elif source == 'bfd':
            if 'tr|' in header:
                species = get_identifiers(header).species_id
                result[species][identity] = (seq, identity, deletion_pct, mutation_pct, insert_pct, raw_len)
            else:
                result[header.strip()][identity] = (seq, identity, deletion_pct, mutation_pct, insert_pct, raw_len)
        elif source == 'mgnify':
            result[header.split('BIOMES=')[1].strip()][identity] = (seq, identity, deletion_pct, mutation_pct, insert_pct, raw_len)
        else:
            raise ValueError
        if n is not None and len(result) >= n:
            break
    return result

class ListIDMapper:
    def __init__(self):
        self.list_to_id = {}
        self.id_to_list = {}
        self.counter = 0

    def _make_id(self):
        return f"complex_{self.counter}"

    def register(self, lst):
        """
        Register a list and get a unique reversible ID.
        """
        key = tuple(lst)  # lists aren’t hashable, so use tuple
        if key in self.list_to_id:
            return self.list_to_id[key]

        # New list → assign new ID
        new_id = self._make_id()
        self.list_to_id[key] = new_id
        self.id_to_list[new_id] = key
        self.counter += 1
        return new_id

    def get_list(self, list_id):
        """
        Get the original list back from its ID.
        """
        return list(self.id_to_list[list_id])
    
    def get_id(self, lst):
        """
        Get ID from list
        """
        key = tuple(lst)
        return self.list_to_id.get(key)
    
    def check_entry(self, lst):
        return tuple(lst) in self.list_to_id

    def save(self, filename):
        """
        Save both mappings to a pickle file.
        """
        with open(filename, "wb") as f:
            pickle.dump({
                "list_to_id": self.list_to_id,
                "id_to_list": self.id_to_list,
                "counter": self.counter
            }, f)

    def load(self, filename):
        """
        Load mappings from a pickle file.
        """
        with open(filename, "rb") as f:
            data = pickle.load(f)
            self.list_to_id = data["list_to_id"]
            self.id_to_list = data["id_to_list"]
            self.counter = data["counter"]

def masked_substring_find(a: str, b: str) -> int:
    n, m = len(a), len(b)
    for i in range(m - n + 1):
        window = b[i:i+n]
        if all(x == y or x == 'X' or y == 'X' for x, y in zip(a, window)):
            return i
    return -1

def remove_x_and_get_mask(s: str):
    mask = [ch != 'X' for ch in s]
    filtered = ''.join([ch for ch, keep in zip(s, mask) if keep])
    return filtered, mask

def map_chain_to_full_local(chain_seq, full_seq):
    """
    Map a chain sequence to a full sequence using local alignment (via edlib).
    Returns the list of indices in full_seq that correspond to each residue in chain_seq.
    """
    # Run local alignment (Smith-Waterman) with CIGAR output
    result = edlib.align(chain_seq, full_seq, mode="HW", task="path")  
    # mode="HW" (semi-global) ensures chain_seq must align end-to-end
    # If you want chain_seq to be a substring inside full_seq, use mode="HW" or "SHW"
    
    start, end = result["locations"][0]  # alignment span in full_seq
    cigar = result["cigar"]

    indices = []
    f_idx = start  # index in full_seq
    c_idx = 0      # index in chain_seq

    num = ""
    for char in cigar:
        if char.isdigit():
            num += char
            continue
        count = int(num) if num else 1
        num = ""

        if char == "=" or char == "X":  # match or mismatch
            for _ in range(count):
                indices.append(f_idx)
                f_idx += 1
                c_idx += 1
        elif char == "I":  # insertion relative to full_seq (extra base in chain)
            c_idx += count
        elif char == "D":  # deletion relative to full_seq (skip in chain)
            f_idx += count
        else:
            raise ValueError(f"Unexpected CIGAR op {char}")

    return indices

def get_pdbs(data_loader, i_worker, data_source, out_dir, repeat=1, max_length=10000, num_units=1000000, consensus_seqs='', msa_match_dict_path='', complex_mapping_path='', msa_dir='', msa_seqs=False, single_species_sample=False, remove_missing=False, id_thresh=0.5, del_thresh=0.2, insrt_thresh=0.2):
    init_alphabet = ['A', 'B', 'C', 'D', 'E', 'F', 'G','H', 'I', 'J','K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T','U', 'V','W','X', 'Y', 'Z', 'a', 'b', 'c', 'd', 'e', 'f', 'g','h', 'i', 'j','k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't','u', 'v','w','x', 'y', 'z']
    extra_alphabet = [str(item) for item in list(np.arange(300))]
    chain_alphabet = init_alphabet + extra_alphabet
    
    # pdb_dict_list = []
    
    out_path = os.path.join(out_dir, str(i_worker) + f'_data_{data_source}.pkl')
    if os.path.exists(out_path):
        return out_path
        # os.remove(out_path)
    if msa_match_dict_path:
        with open(msa_match_dict_path, 'rb') as f:
            msa_match_dict = pickle.load(f)
    else:
        msa_match_dict = None
    if complex_mapping_path and (msa_seqs or consensus_seqs):
        complex_mapper = ListIDMapper()
        complex_mapper.load(complex_mapping_path)
    else:
        complex_mapper = None
    if len(consensus_seqs) > 0:
        if '.json' in consensus_seqs:
            with open(consensus_seqs, 'r') as f:
                consensus_seq_dict = json.load(f)
        else:
            with open(consensus_seqs, 'rb') as f:
                consensus_seq_dict = pickle.load(f)

    def generator():
        c = 0
        c1 = 0
        num_pdb_dicts = 0
        for _ in range(repeat):
            for step,t in enumerate(data_loader):
                if step % 10 == 0:
                    with open(f'{out_dir}/out_log1_{i_worker}.txt', 'a') as f:
                        f.write(str(step) + '\n')
                # with open('/mnt/shared/fosterb/PottsMPNN/log_out1.txt', 'a') as f:
                #     f.write(str(t) + '\n')
                t = {k:v[0] if k not in ['seqs', 'chains'] else [c[0] for c in v] for k,v in t.items()}
                c1 += 1
                if 'label' in list(t):
                    my_dict = {}
                    s = 0
                    concat_seq = ''
                    concat_N = []
                    concat_CA = []
                    concat_C = []
                    concat_O = []
                    concat_mask = []
                    coords_dict = {}
                    mask_list = []
                    visible_list = []
                    mapped_id_list = []
                    if complex_mapper is not None:
                        msa_init_sequence = ''
                        for chain in sorted(list(set(t['chains']))):
                            mapped_id = msa_match_dict.get(t['label'].split('_')[0] + "_" + chain)
                            if mapped_id is None: mapped_id = t['label'].split('_')[0] + "_" + chain
                            mapped_id_list.append(mapped_id)
                            msa_init_sequence += t['seqs'][t['chains'].index(chain)]

                        complex_name = complex_mapper.get_id(mapped_id_list)

                        if complex_name:
                            msa_path = os.path.join(msa_dir, complex_name + '.a3m')
                            msa_seq_stats = parse_a3m_stats(msa_path, dataset='PDB')
                            if 'query' in msa_seq_stats.keys():
                                msa_base_sequence = msa_seq_stats['query'][1.0][0]
                            elif 'query_0' in msa_seq_stats.keys():
                                msa_base_sequence = msa_seq_stats['query_0'][1.0][0]
                            else:
                                try:
                                    msa_base_sequence = list(msa_seq_stats.values())[1.0][0]
                                except Exception as e:
                                    msa_base_sequence = 'X'
                                    msa_init_sequence = 'X'
                            if len(msa_base_sequence) != len(msa_init_sequence):
                                msa_init_sequence = msa_base_sequence

                    if len(list(np.unique(t['idx']))) < 352:
                        aa_num = 0
                        for idx in list(np.unique(t['idx'])):
                            letter = t['chains'][idx]

                            res = np.argwhere(t['idx']==idx).squeeze()
                            if remove_missing:
                                coord_collapse = t['xyz'][:, :4].sum(dim=(-1,-2)).numpy()
                                res_coord = np.argwhere(~np.isnan(coord_collapse)).squeeze()
                                res = np.intersect1d(res, res_coord)
                            initial_sequence = t['seqs'][idx]
                            front_trim = 0
                            back_trim = 0
                            if initial_sequence[4:10] == "HHHHHH":
                                res = res[10:]
                                front_trim = 10
                            elif initial_sequence[3:9] == "HHHHHH":
                                res = res[9:]
                                front_trim = 9
                            elif initial_sequence[2:8] == "HHHHHH":
                                res = res[8:]
                                front_trim = 8
                            elif initial_sequence[1:7] == "HHHHHH":
                                res = res[7:]
                                front_trim = 7
                            elif initial_sequence[0:6] == "HHHHHH":
                                res = res[6:]
                                front_trim = 6
                            if initial_sequence[-10:-4] == "HHHHHH":
                                res = res[:-10]
                                back_trim = -10
                            elif initial_sequence[-9:-3] == "HHHHHH":
                                res = res[:-9]
                                back_trim = -9
                            elif initial_sequence[-8:-2] == "HHHHHH":
                                res = res[:-8]
                                back_trim = -8
                            elif initial_sequence[-7:-1] == "HHHHHH":
                                res = res[:-7]
                                back_trim = -7
                            elif initial_sequence[-6:] == "HHHHHH":
                                res = res[:-6]
                                back_trim = -6
                            if res.shape[0] < 4:
                                pass
                            else:
                                if len(consensus_seqs) > 0:
                                    if msa_match_dict is not None:
                                        cid = msa_match_dict[t['label'].split('_')[0] + '_' + letter]
                                    else:
                                        cid = t['label']
                                    cons_seq = consensus_seq_dict[cid]
                                    if back_trim < 0:
                                        cons_seq = cons_seq[:back_trim]
                                    elif front_trim > 0:
                                        cons_seq = cons_seq[front_trim:]
                                    my_dict['seq_chain_'+letter] = cons_seq
                                    chain_seq = cons_seq

                                elif msa_seqs and ((not complex_mapper) or (complex_mapper and complex_name)):
                                    if 'ingr' in msa_dir or 'ing_msas' in msa_dir:
                                        cid = t['label'].split('_')[0].upper() + '_' + t['label'].split('_')[1]
                                        msa_path = os.path.join(msa_dir, cid + '.a3m')
                                        msa_seq_stats = parse_a3m_stats(msa_path)
                                        base_msa_seq = msa_seq_stats['query'][1.0][0]
                                        if back_trim < 0:
                                            base_msa_seq = base_msa_seq[:back_trim]
                                        if front_trim > 0:
                                            base_msa_seq = base_msa_seq[front_trim:]
                                        use_msa = True
                                        seq_offset = 0
                                        X_mask = []
                                        chain_mapping = []
                                    else:
                                        X_mask = []
                                        chain_mapping = []
                                        seq_offset = msa_init_sequence.find(initial_sequence)
                                        if (msa_init_sequence != 'X') and (seq_offset < 0):
                                            seq_offset = masked_substring_find(initial_sequence, msa_init_sequence)
                                            if seq_offset < 0:
                                                msa_init_sequence_filt, X_mask = remove_x_and_get_mask(msa_init_sequence)
                                                seq_offset = msa_init_sequence_filt.find(initial_sequence)
                                                if seq_offset < 0:
                                                    X_mask = []
                                                    chain_mapping = map_chain_to_full_local(initial_sequence, msa_init_sequence)
                                                    test_mapping = "".join(msa_init_sequence[i] for i in chain_mapping)
                                                    seq_offset = masked_substring_find(initial_sequence, test_mapping)
                                                else:
                                                    msa_init_sequence = msa_init_sequence_filt
                                        use_msa = len(msa_seq_stats.items()) > 0
                                        if seq_offset < 0:
                                            msa_single_seq = t['seqs'][idx]
                                            if back_trim < 0:
                                                msa_single_seq = msa_single_seq[:back_trim]
                                            if front_trim > 0:
                                                msa_single_seq = msa_single_seq[front_trim:]
                                            my_dict['seq_chain_'+letter] = [msa_single_seq] 
                                            chain_seq = my_dict['seq_chain_'+letter][0]
                                            use_msa = False

                                    if use_msa:
                                        filtered = []
                                        for species, species_stats in msa_seq_stats.items():
                                            if single_species_sample:
                                                species_stats = [random.choice(list(species_stats.values()))]
                                            else:
                                                species_stats = list(species_stats.values())
                                            for seq_stats in species_stats:
                                                seq, seq_id, del_pct, mut_pct, insrt_pct, length = seq_stats
                                                if len(X_mask) > 0:
                                                    seq = ''.join([ch for ch, keep in zip(seq, X_mask) if keep])
                                                if len(chain_mapping) > 0:
                                                    seq = "".join(seq[i] for i in chain_mapping)
                                                else:
                                                    seq = seq[seq_offset:seq_offset+len(initial_sequence)]
                                                if seq_id <= id_thresh or del_pct > del_thresh or insrt_pct > insrt_thresh:
                                                    continue
                                                if back_trim < 0:
                                                    seq = seq[:back_trim]
                                                if front_trim > 0:
                                                    seq = seq[front_trim:]
                                                filtered.append(seq)
                                        
                                        my_dict['seq_chain_'+letter] = filtered
                                        my_dict['res_chain_'+letter] = res
                                        chain_seq = my_dict['seq_chain_'+letter][0]

                                elif msa_seqs and complex_mapper:
                                    msa_single_seq = t['seqs'][idx]
                                    if back_trim < 0:
                                        msa_single_seq = msa_single_seq[:back_trim]
                                    if front_trim > 0:
                                        msa_single_seq = msa_single_seq[front_trim:]
                                    my_dict['seq_chain_'+letter] = [msa_single_seq] 
                                    chain_seq = my_dict['seq_chain_'+letter][0]
                                else:
                                    single_seq = t['seqs'][idx]
                                    if back_trim < 0:
                                        single_seq = single_seq[:back_trim]
                                    if front_trim > 0:
                                        single_seq = single_seq[front_trim:]
                                    my_dict['seq_chain_'+letter] = single_seq
                                    chain_seq = my_dict['seq_chain_'+letter]

                                concat_seq += chain_seq
                                if idx in t['masked']:
                                    mask_list.append(letter)
                                else:
                                    visible_list.append(letter)
                                coords_dict_chain = {}

                                all_atoms = np.array(t['xyz'][res,]) #[L, 14, 3]
                                aa_num += all_atoms.shape[0]
                                coords_dict_chain['N_chain_'+letter]=all_atoms[:,0,:].tolist()
                                coords_dict_chain['CA_chain_'+letter]=all_atoms[:,1,:].tolist()
                                coords_dict_chain['C_chain_'+letter]=all_atoms[:,2,:].tolist()
                                coords_dict_chain['O_chain_'+letter]=all_atoms[:,3,:].tolist()
                                my_dict['coords_chain_'+letter]=coords_dict_chain

                        my_dict['name']= t['label']
                        my_dict['masked_list'] = mask_list
                        my_dict['visible_list'] = visible_list
                        my_dict['num_of_chains'] = len(mask_list) + len(visible_list)
                        my_dict['seq'] = concat_seq

                        if len(concat_seq) <= max_length:

                            num_pdb_dicts += 1
                            yield my_dict
                            
                    if num_pdb_dicts >= num_units:
                        break

    # save_in_chunks(generator(), out_path, chunk_size=1000)
    """Save objects in chunks to reduce file I/O overhead."""
    buffer = []
    for obj in generator():
        buffer.append(obj)
        if len(buffer) >= 1000:
            with open(out_path, "ab") as f:
                pickle.dump(buffer, f, protocol=pickle.HIGHEST_PROTOCOL)
            buffer.clear()
    # final flush
    if buffer:
        with open(out_path, "ab") as f:
            pickle.dump(buffer, f, protocol=pickle.HIGHEST_PROTOCOL)
    return out_path

class PDB_dataset(torch.utils.data.Dataset):
    def __init__(self, IDs, loader, train_dict, params):
        self.IDs = IDs
        self.train_dict = train_dict
        self.loader = loader
        self.params = params

    def __len__(self):
        return len(self.IDs)

    def __getitem__(self, index):
        ID = self.IDs[index]
        sel_idx = np.random.randint(0, len(self.train_dict[ID]))
        out = self.loader(self.train_dict[ID][sel_idx], self.params)
        return out

def aa_to_similar(residue):
    if residue == 'MSE':  # convert MSE (seleno-met) to MET
        residue = 'MET'
    elif residue == 'FME':  # convert MSE (n-formylmethionine) to MET
        residue = 'MET'
    elif residue == 'HIC': # convert HIC (4-methyl-histidine) to HIS
        residue = 'HIS'
    elif residue == 'SEP':  # convert SEP (phospho-ser) to SER
        residue = 'SER'
    elif residue == 'SAC':  # convert SAC (n-acetyl-ser) to SER
        residue = 'SER'
    elif residue == 'OAS':  # convert OAS (o-acetyl-ser) to SER
        residue = 'SER'
    elif residue == 'TPO':  # convert TPO (phospho-thr) to THR
        residue = 'THR'
    elif residue == 'IYT':  # convert IYT (n-alpha-acetyl-3,5-diiodotyrosyl-d-threonine) to THR
        residue = 'THR'
    elif residue == 'PTR':  # convert PTR (phospho-tyr) to TYR
        residue = 'TYR'
    elif residue == 'TYS':  # convert TYS (o-sulfo-l-tyr) to TYR
        residue = 'TYR'
    elif residue == 'CSO':  # convert CSO (hydroxy-cys) to CYS
        residue = 'CYS'
    elif residue == 'SEC':  # convert SEC (seleno-cys) to CYS
        residue = 'CYS'
    elif residue == 'CSS':  # convert CSS (s-mercaptocysteine) to CYS
        residue = 'CYS'
    elif residue == 'CAS':  # convert CAS (s-(dimethylarsenic)cysteine) to CYS
        residue = 'CYS'
    elif residue == 'CAF':  # convert CAF (s-dimethylarsinoyl-cysteine) to CYS
        residue = 'CYS'
    elif residue == 'OCS':  # convert OCS (cysteine sulfonic acid) to CYS
        residue = 'CYS'
    elif residue == 'CSD':  # convert CSD (3-sulfinoalanine) to CYS
        residue = 'CYS'
    elif residue == 'CME':  # convert CME (s,s-(2-hydroxyethyl)thiocysteine) to CYS
        residue = 'CYS'
    elif residue == 'YCM':  # convert YCM (s-(2-amino-2-oxoethyl)-l-cysteine) to CYS
        residue = 'CYS'
    elif residue == 'SAH':  # convert SAH (s-adenosyl-l-homocysteine) to CYS
        residue = 'CYS'
    elif residue == 'HYP':  # convert HYP (4-hydroxyproline) to PRO
        residue = 'PRO'
    elif residue == 'M3L':  # convert M3L (n-trimethyllysine) to LYS
        residue = 'LYS'
    elif residue == 'LLP':  # convert LLP (n'-pyridoxyl-lysine-5'-monophosphate) to LYS
        residue = 'LYS'
    elif residue == 'KPI':  # convert KPI ((2s)-2-amino-6-[(1-hydroxy-1-oxo-propan-2-ylidene)amino]hexanoic acid) to LYS
        residue = 'LYS'
    elif residue == 'KPX':  # convert KPX (lysine nz-corboxylic acid) to LYS
        residue = 'LYS'
    elif residue == 'MLY':  # convert MLY (n-dimethyl-lysine) to LYS
        residue = 'LYS'
    elif residue == 'KCX':  # convert KCX (lysine nz-carboxylic acid) to LYS
        residue = 'LYS'
    elif residue == 'PCA':  # convert PCA (pyroglutamic acid) to GLN
        residue = 'GLN'
    elif residue == 'DGL':  # convert DGL (d-glutamic acid) to GLU
        residue = 'GLU'
    elif residue == 'BHD':  # convert BHD (beta-hydroxyaspartic acid) to ASP
        residue = 'ASP'
    elif residue == 'IAS':  # convert IAS (beta-l-aspartic acid) to ASP
        residue = 'ASP'
    elif residue == 'ABA':  # convert ABA (alpha-aminobutyric acid) to ALA
        residue = 'ALA'
    elif residue == '0A9':  # convert 0A9 (methyl l-phenylalaninate) to PHE
        residue = 'PHE'
    return residue

def extract_sequence_and_coords(pdb_file, chain_id='A'):
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure('struct', pdb_file)

    model = structure[0]
    if chain_id not in model:
        raise ValueError(f"Chain {chain_id} not found.")
    chain = model[chain_id]

    coords = []
    sequence = []

    for residue in chain:
        het_flag = residue.id[0] != ' '
        resname = aa_to_similar(residue.get_resname().upper())

        # Skip if not a standard amino acid (either from ATOM or HETATM)
        if het_flag and resname not in STANDARD_AA_RESNAMES:
            continue
        
        if not het_flag and not is_aa(residue, standard=True):
            continue

        try:
            aa = seq1(resname)
        except KeyError:
            aa = 'X'

        atom_coords = []
        skip = False
        for atom_name in ATOM_ORDER:
            if atom_name in residue:
                atom_coords.append(residue[atom_name].get_coord())
            elif atom_name in ['N', 'C', 'CA', 'O']:
                skip = True
                break
            else:
                atom_coords.append([np.nan, np.nan, np.nan])
        if skip:
            continue
        coords.append(np.array(atom_coords))
        sequence.append(aa)

    coords_tensor = torch.tensor(np.array(coords), dtype=torch.float32)  # Shape: L x 14 x 3
    return ''.join(sequence), coords_tensor

def deduplicate_with_indices(lst):
    seen = set()
    deduped = []
    indices = []

    for i, item in enumerate(lst):
        # Convert to tuple to make it hashable
        item_tuple = tuple(item)
        if item_tuple not in seen:
            seen.add(item_tuple)
            deduped.append(item)
            indices.append(i)

    return indices, deduped

def loader_pdb(item,params):
    pdbid,chid = item[0].split('_')
    if params['CATH']:
        seq, xyz = extract_sequence_and_coords(os.path.join(params['DIR'], pdbid.upper() + '_' + chid + '.pdb'), chid)
        idx = torch.zeros(len(seq)).to(dtype=torch.int64)
        xyz = xyz.to(dtype=torch.float32)
        masked = [0]
        return {'seqs'    : [seq],
            'xyz'    : xyz,
            'idx'    : idx,
            'masked' : torch.Tensor(masked).int(),
            'label'  : item[0],
            'chains': ['A']}
    
    PREFIX = "%s/%s/%s"%(params['DIR'],pdbid[1:3],pdbid)
    
    # load metadata
    if not os.path.isfile(PREFIX+".pt"):
        return {'seq': np.zeros(5)}
    meta = torch.load(PREFIX+".pt")
    try:
        asmb_ids, asmb_chains = deduplicate_with_indices(meta['asmb_chains'])
    except:
        asmb_ids = [0]
        asmb_chains = [chid]

    chids = np.array(meta['chains'])
    # find candidate assemblies which contain chid chain
    asmb_candidates = set([a for a,b in zip(asmb_ids,asmb_chains)
                        if chid in b.split(',')])

    # if the chains is missing is missing from all the assemblies
    # then return this chain alone
    if len(asmb_candidates)<1:
        chain = torch.load("%s_%s.pt"%(PREFIX,chid))
        L = len(chain['seq'])

        return {'seqs'    : [chain['seq']],
                'xyz'    : chain['xyz'],
                'idx'    : torch.zeros(L).int(),
                'masked' : torch.Tensor([0]).int(),
                'label'  : item[0],
                'chains' : [chid]}

    # randomly pick one assembly from candidates
    idx = random.sample(list(asmb_candidates), 1)[0]
    xform_idx = idx

    # load relevant chains
    chains = {c:torch.load("%s_%s.pt"%(PREFIX,c))
              for c in meta['asmb_chains'][idx]
              if c in meta['chains']}

    # generate assembly
    asmb = {}

    # pick idx-th xform
    apply_xform = False
    if 'asmb_xform%d'%xform_idx in meta:
        xform = meta['asmb_xform%d'%xform_idx]
        u = xform[:,:3,:3]
        r = xform[:,:3,3]
        apply_xform = True

    # select chains which idx-th xform should be applied to
    s1 = set(meta['chains'])
    s2 = set(meta['asmb_chains'][idx].split(','))
    chains_k = s1&s2

    # transform selected chains 
    for c in chains_k:
        try:
            xyz = chains[c]['xyz']
            if apply_xform:
                xyz_ru = torch.einsum('bij,raj->brai', u, xyz) + r[:,None,None,:]
            else:
                xyz_ru = xyz.unsqueeze(0)
            asmb.update({(c,idx,i):xyz_i for i,xyz_i in enumerate(xyz_ru)})
        except KeyError:
            return {'seq': np.zeros(5)}

    # select chains which share considerable similarity to chid
    # if params['CATH']:
    #     homo = set([chid])
    seqid = meta['tm'][chids==chid][0,:,1]
    homo = set([ch_j for seqid_j,ch_j in zip(seqid,chids)
                if seqid_j>params['HOMO']])
    # stack all chains in the assembly together
    seq,xyz,idx,masked = "",[],[],[]
    seq_list = []
    chain_list = []
    for counter,(k,v) in enumerate(asmb.items()):
        seq += chains[k[0]]['seq']
        seq_list.append(chains[k[0]]['seq'])
        xyz.append(v)
        idx.append(torch.full((v.shape[0],),counter))
        if k[0] in homo:
            masked.append(counter)
        chain_list.append(k[0])

    return {'seqs'    : seq_list,
            'xyz'    : torch.cat(xyz,dim=0),
            'idx'    : torch.cat(idx,dim=0),
            'masked' : torch.Tensor(masked).int(),
            'label'  : item[0],
            'chains' : chain_list}

def build_training_clusters(params, debug):
    val_ids = set([int(l) for l in open(params['VAL']).readlines()])
    test_ids = set([int(l) for l in open(params['TEST']).readlines()])
   
    if debug:
        val_ids = []
        test_ids = []
 
    # read & clean list.csv
    with open(params['LIST'], 'r') as f:
        reader = csv.reader(f)
        next(reader)
        rows = [[r[0],r[3],int(r[4])] for r in reader
                if float(r[2])<=params['RESCUT'] and
                parser.parse(r[1])<=parser.parse(params['DATCUT'])]
    
    # compile training and validation sets
    train = {}
    valid = {}
    test = {}

    if debug:
        rows = rows[:20]
    for r in rows:
        if r[2] in val_ids:
            if r[2] in valid.keys():
                valid[r[2]].append(r[:2])
            else:
                valid[r[2]] = [r[:2]]
        elif r[2] in test_ids:
            if r[2] in test.keys():
                test[r[2]].append(r[:2])
            else:
                test[r[2]] = [r[:2]]
        else:
            if r[2] in train.keys():
                train[r[2]].append(r[:2])
            else:
                train[r[2]] = [r[:2]]
    if debug:
        valid=train       
    return train, valid, test
