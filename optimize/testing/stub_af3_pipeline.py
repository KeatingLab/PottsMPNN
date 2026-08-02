"""A stand-in for ``run_mutation_af3_pipeline.py``.

Accepts the same CLI flags and reproduces the same *observable contract* --
output filename, prepended wildtype row, column names, and AF3 job-directory
layout -- but fabricates metrics from a deterministic hash instead of running
AlphaFold3, ipSAE and PISA. This lets the whole loop be exercised without GPUs
or cluster access.

Contract reproduced (see run_mutation_af3_pipeline.py):

* writes ``<out_dir>/<input_csv_stem>_with_af3.csv``
* prepends a ``mutations == "WT"`` row
* adds ``ipSAE``, ``int_area``, ``dG_binding``, ``dG_diss``
* creates ``<out_dir>/<job_name>/seed-1_sample-N/*_model.cif`` plus the
  ``*_model_<pae>_<dist>.txt`` ipSAE report beside each model
"""

from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path

import pandas as pd

N_SAMPLES = 3
BACKBONE_OFFSETS = {"N": -0.5, "CA": 0.0, "C": 0.6, "O": 1.1}
ONE_TO_THREE = {
    "A": "ALA", "R": "ARG", "N": "ASN", "D": "ASP", "C": "CYS",
    "Q": "GLN", "E": "GLU", "G": "GLY", "H": "HIS", "I": "ILE",
    "L": "LEU", "K": "LYS", "M": "MET", "F": "PHE", "P": "PRO",
    "S": "SER", "T": "THR", "W": "TRP", "Y": "TYR", "V": "VAL",
}


def _unit_hash(text: str, salt: str) -> float:
    digest = hashlib.sha256(f"{salt}:{text}".encode("utf-8")).hexdigest()
    return int(digest[:8], 16) / 0xFFFFFFFF


def _count_mutations(mutations) -> int:
    """Mutation count, treating NaN/empty/'WT' as the wildtype.

    An empty ``mutations`` string round-trips through CSV as NaN.
    """
    if mutations is None:
        return 0
    text = str(mutations).strip()
    if text == "" or text.lower() == "nan" or text == "WT":
        return 0
    return len([m for m in text.split(",") if m.strip()])


def _metrics(key: str, mutations, drift: float) -> dict:
    """Fabricate the pipeline's four output metrics.

    Directions match the real ones: ipSAE and dG_diss are better high,
    dG_binding (interface solvation energy) is better low.
    """
    n = _count_mutations(mutations)
    return {
        "ipSAE": round(0.50 + drift * n * 0.05 + _unit_hash(key, "ipsae") * 0.10, 4),
        "int_area": round(800.0 + _unit_hash(key, "area") * 400.0, 2),
        "dG_binding": round(-10.0 - drift * n * 1.0 - _unit_hash(key, "solv") * 3.0, 4),
        "dG_diss": round(5.0 + drift * n * 1.5 + _unit_hash(key, "diss") * 2.0, 4),
    }


# ----------------------------------------------------- AF3 layout replication


def _safe_token(text: str) -> str:
    return text.replace(":", "_").replace(",", "_").replace(" ", "")


def _af3_tokens(mutations) -> list:
    text = str(mutations).strip()
    if text in ("", "nan", "WT"):
        return ["WT"]
    tokens = []
    for tok in text.split(","):
        tok = tok.strip()
        if not tok:
            continue
        chain, rest = tok.split(":")
        tokens.append(f"{chain}{rest}")
    return tokens or ["WT"]


def _parse_fasta(path: str):
    with open(path, encoding="utf-8") as handle:
        lines = [l.strip() for l in handle if l.strip()]
    header = lines[0][1:]
    parts = header.split("|")
    chain_ids = parts[1].split(":") if len(parts) > 1 else ["A"]
    return parts[0], dict(zip(chain_ids, lines[1].split(":")))


def _apply(chain_seqs: dict, mutations) -> dict:
    out = {c: list(s) for c, s in chain_seqs.items()}
    text = str(mutations).strip()
    if text in ("", "nan", "WT"):
        return {c: "".join(s) for c, s in out.items()}
    for tok in text.split(","):
        tok = tok.strip()
        if not tok:
            continue
        chain, rest = tok.split(":")
        pos = int(rest[1:-1])
        out[chain][pos - 1] = rest[-1]
    return {c: "".join(s) for c, s in out.items()}


def _write_fake_model(path: Path, chain_seqs: dict) -> None:
    """Write a minimal mmCIF encoding the mutant sequence on a dummy backbone."""
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    serial = 1
    for chain, seq in chain_seqs.items():
        for index, residue in enumerate(seq, start=1):
            for atom, offset in BACKBONE_OFFSETS.items():
                rows.append(
                    f"ATOM {serial} {atom} . {ONE_TO_THREE.get(residue, 'UNK')} {chain} "
                    f"{index} {index * 3.8 + offset:.3f} 0.000 0.000 1"
                )
                serial += 1
    header = (
        "data_stub\nloop_\n"
        "_atom_site.group_PDB\n_atom_site.id\n_atom_site.label_atom_id\n"
        "_atom_site.label_alt_id\n_atom_site.label_comp_id\n_atom_site.auth_asym_id\n"
        "_atom_site.auth_seq_id\n_atom_site.Cartn_x\n_atom_site.Cartn_y\n"
        "_atom_site.Cartn_z\n_atom_site.pdbx_PDB_model_num\n"
    )
    path.write_text(header + "\n".join(rows) + "\n#\n", encoding="utf-8")


def _write_ipsae_report(path: Path, value: float) -> None:
    """Whitespace table with the Type/ipSAE columns the pipeline reads."""
    path.write_text(
        "Chn1 Chn2 Type ipSAE\n"
        f"A B asym {value:.4f}\n"
        f"B A asym {value + 0.01:.4f}\n"
        f"A B max  {value + 0.02:.4f}\n",
        encoding="utf-8",
    )


def _emit_af3_outputs(out_dir: Path, base_name: str, mutations, chain_seqs: dict,
                      ipsae: float, pae_cutoff: int, dist_cutoff: int) -> None:
    job_name = _safe_token(f"{base_name}__{'_'.join(_af3_tokens(mutations))}").lower()
    job_dir = out_dir / job_name
    n_tokens = sum(len(s) for s in chain_seqs.values())
    for sample in range(N_SAMPLES):
        sample_dir = job_dir / f"seed-1_sample-{sample}"
        model = sample_dir / f"{job_name}_model.cif"
        _write_fake_model(model, chain_seqs)
        stem = str(model).replace(".cif", "")
        # Sample 0 is made the best, so model selection is deterministic.
        score = ipsae if sample == 0 else ipsae - 0.05 * sample
        _write_ipsae_report(Path(f"{stem}_{pae_cutoff:02d}_{dist_cutoff:02d}.txt"), score)

        # The byproducts the real pipeline leaves behind, coarsened: the cleanup
        # stage targets the O(tokens^2) PAE matrix.
        row = ",".join("0.0" for _ in range(min(n_tokens, 64)))
        Path(f"{stem}_confidences.json").write_text(
            '{"pae": [' + ",".join(f"[{row}]" for _ in range(min(n_tokens, 64))) + "]}",
            encoding="utf-8",
        )
        Path(f"{stem}_summary_confidences.json").write_text(
            f'{{"iptm": {score:.4f}, "ptm": {score:.4f}}}', encoding="utf-8"
        )
        Path(f"{stem}_pisa.xml").write_text(
            "<pisa><INTERFACE><int_area>900.0</int_area>"
            "<int_solv_en>-12.0</int_solv_en></INTERFACE></pisa>", encoding="utf-8"
        )
        Path(f"{stem}_pisa_assemblies.xml").write_text(
            "<pisa><assembly><size>2</size><diss_energy>8.0</diss_energy>"
            "</assembly></pisa>", encoding="utf-8"
        )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mutations_csv", required=True)
    ap.add_argument("--fasta", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--msa_json")
    ap.add_argument("--pisa_exe")
    ap.add_argument("--pisa_cfg")
    ap.add_argument("--pisa_name")
    ap.add_argument("--ipsae_script")
    ap.add_argument("--pae_cutoff", type=int, default=10)
    ap.add_argument("--dist_cutoff", type=int, default=15)
    ap.add_argument("--max_parallel", type=int, default=1)
    # Stub-only knobs.
    ap.add_argument("--drift", type=float, default=1.0,
                    help="How strongly extra mutations improve the fabricated metrics.")
    ap.add_argument("--emit_structures", action="store_true",
                    help="Also write fake AF3 job directories with models and ipSAE reports.")
    ap.add_argument("--fail", action="store_true",
                    help="Exit non-zero, to exercise the loop's error handling.")
    args = ap.parse_args()

    if args.fail:
        raise SystemExit("stub pipeline asked to fail")

    frame = pd.read_csv(args.mutations_csv)
    if "mutations" not in frame.columns:
        raise SystemExit(f"Expected 'mutations' column; got {list(frame.columns)}")

    base_name, base_chain_seqs = _parse_fasta(args.fasta)

    # Prepend the wildtype row, exactly as the real pipeline does.
    score_cols = [c for c in ("stability_score", "binding_score", "uniqueness_score")
                  if c in frame.columns]
    wt_row = {"mutations": "WT", **{c: 0.0 for c in score_cols}}
    frame = pd.concat([pd.DataFrame([wt_row]), frame], ignore_index=True)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    computed = [_metrics(str(m), m, args.drift) for m in frame["mutations"]]
    for column in ("ipSAE", "int_area", "dG_binding", "dG_diss"):
        frame[column] = [c[column] for c in computed]

    if args.emit_structures:
        for mutations, metrics in zip(frame["mutations"], computed):
            _emit_af3_outputs(
                out_dir, base_name, mutations, _apply(base_chain_seqs, mutations),
                metrics["ipSAE"], args.pae_cutoff, args.dist_cutoff,
            )

    target = out_dir / f"{Path(args.mutations_csv).stem}_with_af3.csv"
    frame.to_csv(target, index=False)
    print(f"stub pipeline wrote {len(frame)} row(s) to {target}")


if __name__ == "__main__":
    main()
