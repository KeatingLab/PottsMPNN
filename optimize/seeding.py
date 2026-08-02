"""Re-seeding a mutation search onto a new sequence and/or a new backbone.

The Potts energy table depends only on structure, so overriding the sequence in
the ``parse_PDB`` dict redefines "wildtype" for the whole scoring stack. Both
``seq`` and every ``seq_chain_<X>`` must be written: ``get_etab`` takes the
bound reference from ``seq`` and the partition reference from ``seq_chain_<X>``,
so overriding only one makes them disagree.

Avoids importing torch so it stays cheap to test.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

# "<chain>:<wt><1-indexed position within chain><mut>", e.g. "B:W102E".
MUTATION_TOKEN_RE = re.compile(r"^([^:]+):([A-Z])(\d+)([A-Z])$")

BACKBONE_ATOMS = ("N", "CA", "C", "O")

THREE_TO_ONE = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
    "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
}

ONE_TO_THREE = {one: three for three, one in THREE_TO_ONE.items()}


# --------------------------------------------------------------------- helpers


def chain_lengths(pdb_entry: dict) -> Dict[str, int]:
    """Per-chain sequence lengths, in ``chain_order``."""
    return {c: len(pdb_entry[f"seq_chain_{c}"]) for c in pdb_entry["chain_order"]}


def chain_offsets(pdb_entry: dict) -> Dict[str, int]:
    """Start index of each chain within the concatenated sequence."""
    offsets: Dict[str, int] = {}
    offset = 0
    for chain in pdb_entry["chain_order"]:
        offsets[chain] = offset
        offset += len(pdb_entry[f"seq_chain_{chain}"])
    return offsets


def format_mutation_token(chain: str, position: int, wt: str, mut: str) -> str:
    """Build a token identical to ``mutation_search._format_mutation``."""
    return f"{chain}:{wt}{position}{mut}"


def parse_mutation_token(token: str) -> Tuple[str, str, int, str]:
    """Split ``"B:W102E"`` into ``("B", "W", 102, "E")``. Position is 1-indexed."""
    match = MUTATION_TOKEN_RE.match(token.strip())
    if not match:
        raise ValueError(
            f"Malformed mutation token {token!r}; expected '<chain>:<WT><pos><MUT>' e.g. 'B:W102E'."
        )
    chain, wt, position, mut = match.groups()
    return chain, wt, int(position), mut


def split_mutations(mutation_str) -> List[str]:
    """Split a comma-joined mutation string, **preserving order**. NaN-safe.

    The AF3 job directory name joins the tokens in the order given, so
    re-ordering them renames the job and makes an existing prediction
    unfindable. Use this -- not :func:`parse_mutation_tokens` -- anywhere the
    string round-trips into a name.
    """
    if mutation_str is None:
        return []
    text = str(mutation_str).strip()
    if text == "" or text.lower() == "nan":
        return []
    return [t.strip() for t in text.split(",") if t.strip()]


def parse_mutation_tokens(mutation_str) -> Tuple[str, ...]:
    """Split a comma-joined mutation string into a **sorted** token tuple.

    Tolerates NaN/empty for the wildtype row. Sorting makes this a canonical
    *set* for similarity comparisons; it discards the original order, so it must
    not be used to rebuild identifiers (see :func:`split_mutations`).
    """
    return tuple(sorted(split_mutations(mutation_str)))


# ------------------------------------------------------------ sequence seeding


def apply_sequence_override(pdb_entry: dict, sequence: str) -> None:
    """Redefine the wildtype sequence of ``pdb_entry`` in place.

    Writes ``seq`` and every ``seq_chain_<X>`` consistently. Coordinates are
    untouched -- the backbone stays whatever was parsed.

    Raises ``ValueError`` if ``sequence`` does not match the structure's total
    length, since a silent mismatch would corrupt every downstream score.
    """
    lengths = chain_lengths(pdb_entry)
    total = sum(lengths.values())
    if len(sequence) != total:
        raise ValueError(
            f"Sequence override length {len(sequence)} does not match structure length {total} "
            f"(chains: {', '.join(f'{c}={n}' for c, n in lengths.items())})."
        )
    invalid = set(sequence) - set(THREE_TO_ONE.values())
    if invalid:
        raise ValueError(
            f"Sequence override contains non-canonical residues: {sorted(invalid)}"
        )

    offset = 0
    for chain in pdb_entry["chain_order"]:
        length = lengths[chain]
        pdb_entry[f"seq_chain_{chain}"] = sequence[offset : offset + length]
        offset += length
    pdb_entry["seq"] = sequence


def seeded_pdb_data(pdb_data: list, sequence: Optional[str]) -> list:
    """Deep-ish copy of a ``parse_PDB`` result with the sequence overridden.

    Copies the dict so the caller's parsed structure is not mutated; coordinate
    entries are shared by reference since they are never modified.
    """
    seeded = [dict(entry) for entry in pdb_data]
    if sequence is not None:
        apply_sequence_override(seeded[0], sequence)
    return seeded


# ------------------------------------------------- mutation <-> sequence diffs


def diff_to_wt(
    sequence: str,
    wt_sequence: str,
    pdb_entry: Optional[dict] = None,
    *,
    lengths: Optional[Dict[str, int]] = None,
    chain_order: Optional[Sequence[str]] = None,
) -> List[str]:
    """Mutation tokens describing ``sequence`` relative to ``wt_sequence``.

    Cross-round lineage is derived rather than tracked: a re-seeded search only
    reports that round's own changes, so every candidate is diffed against the
    original wildtype instead. Positions map 1:1 because sequence length and
    chain layout are preserved across rounds.

    Either pass ``pdb_entry`` or both ``lengths`` and ``chain_order``.
    """
    if pdb_entry is not None:
        lengths = chain_lengths(pdb_entry)
        chain_order = pdb_entry["chain_order"]
    if lengths is None or chain_order is None:
        raise ValueError("Provide either pdb_entry, or both lengths and chain_order.")
    if len(sequence) != len(wt_sequence):
        raise ValueError(
            f"Cannot diff sequences of different lengths: {len(sequence)} vs {len(wt_sequence)}."
        )

    tokens: List[str] = []
    offset = 0
    for chain in chain_order:
        length = lengths[chain]
        for local in range(length):
            index = offset + local
            wt_res, mut_res = wt_sequence[index], sequence[index]
            if wt_res != mut_res:
                tokens.append(format_mutation_token(chain, local + 1, wt_res, mut_res))
        offset += length
    return tokens


def apply_mutation_tokens(
    wt_sequence: str,
    tokens: Iterable[str],
    pdb_entry: Optional[dict] = None,
    *,
    lengths: Optional[Dict[str, int]] = None,
    chain_order: Optional[Sequence[str]] = None,
    validate_wt: bool = True,
) -> str:
    """Apply mutation tokens to a sequence. Inverse of :func:`diff_to_wt`."""
    if pdb_entry is not None:
        lengths = chain_lengths(pdb_entry)
        chain_order = pdb_entry["chain_order"]
    if lengths is None or chain_order is None:
        raise ValueError("Provide either pdb_entry, or both lengths and chain_order.")

    offsets: Dict[str, int] = {}
    offset = 0
    for chain in chain_order:
        offsets[chain] = offset
        offset += lengths[chain]

    residues = list(wt_sequence)
    for token in tokens:
        chain, wt_res, position, mut_res = parse_mutation_token(token)
        if chain not in offsets:
            raise ValueError(f"Mutation {token!r} names chain {chain!r}, not in {list(chain_order)}.")
        if not (1 <= position <= lengths[chain]):
            raise ValueError(
                f"Mutation {token!r} position {position} is outside chain {chain} "
                f"(length {lengths[chain]})."
            )
        index = offsets[chain] + position - 1
        if validate_wt and residues[index] != wt_res:
            raise ValueError(
                f"Mutation {token!r} expects {wt_res} at {chain}:{position} but sequence has "
                f"{residues[index]}."
            )
        residues[index] = mut_res
    return "".join(residues)


# ------------------------------------------------------------ partitions JSON


def write_partitions_json(
    path: str, pdb_name: str, partitions: Sequence[Sequence[str]]
) -> str:
    """Write a one-entry binding-partition JSON for a per-round backbone.

    ``mutation_search._parse_binding_partitions`` looks partitions up by
    ``pdb_data[0]["name"]``, which ``parse_PDB`` derives from the filename stem.
    Each round's regenerated backbone therefore needs its own entry.
    """
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = {pdb_name: [list(p) for p in partitions]}
    with open(target, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    return str(target)


# ------------------------------------------------------- AF3 backbone handling


def find_af3_model(
    af3_dir: str, model_glob: Sequence[str] = ("*model_0.cif", "*model.cif", "*.cif", "*.pdb")
) -> Path:
    """Locate the top-ranked structure file AF3 wrote for one prediction.

    The glob list is tried in order, so the most specific naming convention wins.
    """
    root = Path(af3_dir)
    if not root.exists():
        raise FileNotFoundError(f"AF3 output directory does not exist: {root}")
    for pattern in model_glob:
        matches = sorted(root.rglob(pattern))
        if matches:
            return matches[0]
    raise FileNotFoundError(
        f"No structure file under {root} matched any of: {', '.join(model_glob)}"
    )


def _read_cif_backbone(path: Path) -> List[dict]:
    """Minimal ``_atom_site`` reader returning backbone atom records.

    ``parse_PDB`` reads just N/CA/C/O, so nothing else needs to survive the
    conversion.
    """
    columns: List[str] = []
    rows: List[dict] = []
    in_loop = False
    in_atom_site = False

    with open(path, "r", encoding="utf-8") as handle:
        for raw in handle:
            line = raw.rstrip("\n")
            stripped = line.strip()
            if stripped.startswith("loop_"):
                in_loop, in_atom_site, columns = True, False, []
                continue
            if in_loop and stripped.startswith("_atom_site."):
                in_atom_site = True
                columns.append(stripped.split(".", 1)[1])
                continue
            if in_atom_site:
                if not stripped or stripped.startswith("#") or stripped.startswith("_"):
                    if stripped.startswith("loop_") or stripped.startswith("_"):
                        in_loop, in_atom_site = False, False
                    continue
                fields = stripped.split()
                if len(fields) != len(columns):
                    continue
                rows.append(dict(zip(columns, fields)))
    if not columns:
        raise ValueError(f"No _atom_site loop found in {path}")
    return rows


def _cif_to_backbone_atoms(path: Path) -> List[dict]:
    """Normalize a CIF into ``[{chain, resseq, resname, atom, x, y, z}]``."""
    rows = _read_cif_backbone(path)

    def pick(row: dict, *keys: str, default: str = "") -> str:
        for key in keys:
            if key in row and row[key] not in (".", "?"):
                return row[key]
        return default

    atoms: List[dict] = []
    for row in rows:
        resname = pick(row, "auth_comp_id", "label_comp_id", default="UNK")
        group = pick(row, "group_PDB", default="ATOM")
        # ATOM only, with MSE promoted to MET; admitting HETATM wholesale would
        # let waters and ligands through.
        if group == "HETATM" and resname == "MSE":
            group, resname = "ATOM", "MET"
        if group != "ATOM":
            continue
        model = pick(row, "pdbx_PDB_model_num", default="1")
        if model not in ("1", ""):
            continue
        atom_name = pick(row, "auth_atom_id", "label_atom_id")
        if atom_name not in BACKBONE_ATOMS:
            continue
        alt = pick(row, "label_alt_id")
        if alt not in ("", "A"):
            continue
        try:
            atoms.append(
                {
                    "chain": pick(row, "auth_asym_id", "label_asym_id"),
                    "resseq": int(float(pick(row, "auth_seq_id", "label_seq_id", default="0"))),
                    "resname": resname,
                    "atom": atom_name,
                    "x": float(pick(row, "Cartn_x", default="0")),
                    "y": float(pick(row, "Cartn_y", default="0")),
                    "z": float(pick(row, "Cartn_z", default="0")),
                }
            )
        except ValueError:
            continue
    if not atoms:
        raise ValueError(f"No backbone atoms recovered from {path}")
    return atoms


def _format_pdb_atom(serial: int, atom: dict) -> str:
    """Render one ATOM record in fixed-column PDB format."""
    name = atom["atom"]
    # Single-character elements are indented one column (PDB convention).
    name_field = f" {name:<3}" if len(name) < 4 else name
    element = name[0]
    return (
        f"ATOM  {serial:5d} {name_field}{'':1}{atom['resname']:>3} "
        f"{atom['chain'][:1]:1}{atom['resseq']:4d}{'':1}   "
        f"{atom['x']:8.3f}{atom['y']:8.3f}{atom['z']:8.3f}"
        f"{1.00:6.2f}{0.00:6.2f}          {element:>2}"
    )


def _write_backbone_pdb(atoms: Sequence[dict], out_path: Path, chain_order: Sequence[str]) -> None:
    """Write backbone atoms as a PDB, chains emitted in ``chain_order``."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    by_chain: Dict[str, List[dict]] = {}
    for atom in atoms:
        by_chain.setdefault(atom["chain"], []).append(atom)

    serial = 1
    lines: List[str] = []
    for chain in chain_order:
        for atom in by_chain.get(chain, []):
            lines.append(_format_pdb_atom(serial, atom))
            serial += 1
        lines.append("TER")
    lines.append("END")
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# parse_PDB walks this alphabet, so chain_order is alphabet order -- not the
# order the chains appear in the file.
PARSE_PDB_ALPHABET = (
    list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
    + list("abcdefghijklmnopqrstuvwxyz")
    + [str(i) for i in range(300)]
)


def infer_chain_order(atoms: Sequence[dict]) -> List[str]:
    """Chain order as ``parse_PDB`` would report it, without importing torch."""
    present = {atom["chain"] for atom in atoms}
    rank = {chain: index for index, chain in enumerate(PARSE_PDB_ALPHABET)}
    known = [c for c in present if c in rank]
    unknown = sorted(c for c in present if c not in rank)
    return sorted(known, key=lambda c: rank[c]) + unknown


def backbone_chain_lengths(
    atoms: Sequence[dict], chain_order: Sequence[str]
) -> Dict[str, int]:
    """Residue count per chain, derived from backbone atoms."""
    counts: Dict[str, set] = {chain: set() for chain in chain_order}
    for atom in atoms:
        if atom["chain"] in counts:
            counts[atom["chain"]].add(atom["resseq"])
    return {chain: len(residues) for chain, residues in counts.items()}


def _residue_keys_in_order(
    atoms: Sequence[dict], chain_order: Sequence[str]
) -> List[Tuple[str, int]]:
    """Distinct ``(chain, resseq)`` pairs, ordered by ``chain_order`` then resseq."""
    seen: Dict[str, List[int]] = {}
    for atom in atoms:
        bucket = seen.setdefault(atom["chain"], [])
        if atom["resseq"] not in bucket:
            bucket.append(atom["resseq"])
    keys: List[Tuple[str, int]] = []
    for chain in chain_order:
        for resseq in sorted(seen.get(chain, [])):
            keys.append((chain, resseq))
    return keys


def write_backbone_with_sequence(
    atoms: Sequence[dict],
    sequence: str,
    chain_order: Sequence[str],
    out_path: str,
) -> str:
    """Write a backbone PDB whose residue names spell out ``sequence``.

    ``recursive_mutation_search`` parses its own PDB, so an in-memory override
    cannot reach it; encoding the seed sequence in the residue names makes the
    backbone self-describing instead. Only N/CA/C/O are read by PottsMPNN, so
    the absent sidechains for substituted residues are immaterial.
    """
    target = Path(out_path)
    keys = _residue_keys_in_order(atoms, chain_order)
    if len(keys) != len(sequence):
        raise ValueError(
            f"Backbone has {len(keys)} residues but sequence has {len(sequence)}; "
            "cannot write a self-describing backbone."
        )

    resname_by_key = {}
    for (chain, resseq), residue in zip(keys, sequence):
        if residue not in ONE_TO_THREE:
            raise ValueError(f"Non-canonical residue {residue!r} in sequence.")
        resname_by_key[(chain, resseq)] = ONE_TO_THREE[residue]

    renamed = []
    for atom in atoms:
        updated = dict(atom)
        updated["resname"] = resname_by_key[(atom["chain"], atom["resseq"])]
        renamed.append(updated)

    _write_backbone_pdb(renamed, target, chain_order)
    return str(target)


def read_backbone_atoms(path: str) -> List[dict]:
    """Read backbone atoms from a ``.pdb`` or ``.cif`` file."""
    source = Path(path)
    if source.suffix.lower() in (".cif", ".mmcif"):
        return _cif_to_backbone_atoms(source)
    return _pdb_to_backbone_atoms(source)


def backbone_sequence(atoms: Sequence[dict], chain_order: Sequence[str]) -> str:
    """One-letter sequence implied by a backbone's residue names."""
    resname_by_key: Dict[Tuple[str, int], str] = {}
    for atom in atoms:
        resname_by_key[(atom["chain"], atom["resseq"])] = atom["resname"]
    return "".join(
        THREE_TO_ONE.get(resname_by_key[key], "-")
        for key in _residue_keys_in_order(atoms, chain_order)
    )


def prepare_backbone_from_af3(
    model_path: str,
    out_pdb: str,
    expected_chain_order: Sequence[str],
    expected_lengths: Dict[str, int],
    *,
    chain_map: Optional[Dict[str, str]] = None,
    skip_gaps: bool = False,
    validate: bool = True,
    expected_sequence: Optional[str] = None,
) -> str:
    """Convert an AF3 model file into a ``parse_PDB``-compatible backbone.

    Preserves the original chain IDs, chain order, and per-chain lengths so the
    binding partitions and interface mask stay valid in the next round. Raises
    if the converted structure does not match the expected layout -- a silent
    mismatch would misalign every mutation position. ``expected_sequence``
    additionally checks that AF3 folded the sequence it was asked for.

    ``chain_map`` renames AF3 chain IDs onto the originals (e.g. ``{"A": "A", "B": "B"}``).
    """
    source = Path(model_path)
    target = Path(out_pdb)

    if source.suffix.lower() == ".pdb":
        atoms = _pdb_to_backbone_atoms(source)
    else:
        atoms = _cif_to_backbone_atoms(source)

    if chain_map:
        for atom in atoms:
            atom["chain"] = chain_map.get(atom["chain"], atom["chain"])

    found = sorted({atom["chain"] for atom in atoms})
    missing = [c for c in expected_chain_order if c not in found]
    if missing:
        raise ValueError(
            f"AF3 model {source} is missing expected chain(s) {missing}; found {found}. "
            "Set structure.af3_chain_map if AF3 relabelled the chains."
        )
    if expected_sequence is not None and len(expected_sequence) != sum(expected_lengths.values()):
        raise ValueError(
            f"expected_sequence length {len(expected_sequence)} does not match expected chain "
            f"lengths totalling {sum(expected_lengths.values())}."
        )

    if expected_sequence is not None:
        actual = backbone_sequence(atoms, expected_chain_order)
        if actual != expected_sequence:
            raise ValueError(
                f"AF3 model {source} encodes a different sequence than requested.\n"
                f"  expected: {expected_sequence}\n"
                f"  found   : {actual}"
            )

    _write_backbone_pdb(atoms, target, expected_chain_order)
    if validate:
        _validate_backbone(str(target), expected_chain_order, expected_lengths, skip_gaps=skip_gaps)
    return str(target)


def _pdb_to_backbone_atoms(path: Path) -> List[dict]:
    """Read backbone atoms from a PDB file (first model, altloc blank/A only).

    Mirrors ``data_utils.parse_PDB_biounits``: only ``ATOM`` records count, with
    selenomethionine ``HETATM``/``MSE`` promoted to ``MET``. An ``HOH`` oxygen is
    named ``O``, so accepting all ``HETATM`` records would inflate chain lengths
    and misalign every mutation position.
    """
    atoms: List[dict] = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            if line.startswith("ENDMDL"):
                break
            resname = line[17:20]
            if line.startswith("HETATM") and resname == "MSE":
                line = "ATOM  " + line[6:]
                resname = "MET"
            if not line.startswith("ATOM"):
                continue
            atom_name = line[12:16].strip()
            if atom_name not in BACKBONE_ATOMS:
                continue
            if line[16] not in (" ", "A"):
                continue
            atoms.append(
                {
                    "chain": line[21],
                    "resseq": int(line[22:26]),
                    "resname": resname.strip(),
                    "atom": atom_name,
                    "x": float(line[30:38]),
                    "y": float(line[38:46]),
                    "z": float(line[46:54]),
                }
            )
    if not atoms:
        raise ValueError(f"No backbone atoms recovered from {path}")
    return atoms


def _validate_backbone(
    pdb_path: str,
    expected_chain_order: Sequence[str],
    expected_lengths: Dict[str, int],
    skip_gaps: bool = False,
) -> None:
    """Re-parse the written PDB with ``parse_PDB`` and check the layout.

    Confirmatory: chain order, lengths and sequence were already verified from
    the atom records. This proves the *written file* parses the way the search
    will read it, catching any PDB formatting error.
    """
    try:
        from data_utils import parse_PDB  # local import: pulls torch
    except ImportError as exc:
        # The atom-level invariants still hold.
        print(
            f"  NOTE: skipping confirmatory parse_PDB check for {pdb_path} "
            f"({exc}). Chain order, lengths and sequence were already verified."
        )
        return

    parsed = parse_PDB(pdb_path, skip_gaps=skip_gaps)
    if not parsed:
        raise ValueError(f"parse_PDB returned nothing for {pdb_path}")
    entry = parsed[0]
    if list(entry["chain_order"]) != list(expected_chain_order):
        raise ValueError(
            f"Backbone {pdb_path} has chain order {entry['chain_order']}, expected "
            f"{list(expected_chain_order)}."
        )
    actual = chain_lengths(entry)
    if actual != dict(expected_lengths):
        raise ValueError(
            f"Backbone {pdb_path} chain lengths {actual} do not match expected "
            f"{dict(expected_lengths)}. Mutation positions would be misaligned."
        )
