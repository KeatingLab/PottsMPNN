"""
Mutate -> AF3 (with provided MSAs) -> ipSAE -> PISA dG_binding pipeline.

Reads a CSV of mutation candidates, applies them to a base FASTA, predicts each
complex with AlphaFold3 using prespecified MSAs, picks the seed with the best
min-asym ipSAE, runs PISA on that structure, and writes an augmented CSV.
"""

import argparse
import json
import math
import multiprocessing as mp
import os
import re
import shutil
import subprocess
import sys
import xml.etree.ElementTree as ET
from glob import glob
from pathlib import Path

import pandas as pd

MUT_RE = re.compile(r"^([A-Za-z0-9]+):([A-Z])(\d+)([A-Z])$")


def parse_fasta(fasta_path):
    with open(fasta_path) as f:
        lines = [l.strip() for l in f if l.strip()]
    if len(lines) < 2 or not lines[0].startswith(">"):
        raise ValueError(f"Bad FASTA: {fasta_path}")
    header = lines[0][1:]
    seq_line = lines[1]
    parts = header.split("|")
    name = parts[0]
    chain_ids = parts[1].split(":") if len(parts) > 1 else ["A"]
    seqs = seq_line.split(":")
    if len(seqs) != len(chain_ids):
        raise ValueError(
            f"Chain id count ({len(chain_ids)}) != seq count ({len(seqs)}) in {fasta_path}"
        )
    return name, dict(zip(chain_ids, seqs))


def apply_mutations(chain_seqs, mut_string):
    """Returns (new_chain_seqs, mut_tokens) or raises ValueError on WT mismatch."""
    new_seqs = {c: list(s) for c, s in chain_seqs.items()}
    tokens = []
    for mut in [m.strip() for m in mut_string.split(",") if m.strip()]:
        m = MUT_RE.match(mut)
        if not m:
            raise ValueError(f"Cannot parse mutation: {mut}")
        chain, wt, resnum_s, mt = m.group(1), m.group(2), m.group(3), m.group(4)
        resnum = int(resnum_s)
        if chain not in new_seqs:
            raise ValueError(f"Chain {chain} not in FASTA")
        idx = resnum - 1
        if idx < 0 or idx >= len(new_seqs[chain]):
            raise ValueError(f"Resnum {resnum} out of range for chain {chain}")
        cur = new_seqs[chain][idx]
        if cur != wt:
            raise ValueError(
                f"WT mismatch at {chain}:{resnum}: expected {wt}, found {cur}"
            )
        new_seqs[chain][idx] = mt
        tokens.append(f"{chain}{wt}{resnum}{mt}")
    return {c: "".join(s) for c, s in new_seqs.items()}, tokens


def load_msa_map(msa_json_path):
    """
    Load chain -> {unpaired, paired, templates} mapping from a JSON file.

    Expected format:
        {
          "A": {
            "unpaired": "/abs/path/to/chain_a_unpaired.a3m",
            "paired":   "/abs/path/to/chain_a_paired.a3m",
            "templates": [                       # optional
              {"mmcifPath": "/abs/path/to/template.cif",
               "queryIndices":    [0, 1, 2, ...],
               "templateIndices": [0, 1, 2, ...]}
            ]
          },
          "B": {"unpaired": "...", "paired": "..."}
        }
    Any key may be omitted; empty/absent unpaired or paired => blank MSA field;
    omitted/empty templates => no templates for that chain.
    """
    with open(msa_json_path) as f:
        raw = json.load(f)
    msa_map = {}
    for chain, entry in raw.items():
        unpaired = str(entry.get("unpaired", "") or "")
        paired   = str(entry.get("paired",   "") or "")
        if unpaired:
            unpaired = os.path.abspath(unpaired)
            if not os.path.exists(unpaired):
                print(f"  WARNING: unpaired MSA for chain {chain} not found: {unpaired}")
        if paired:
            paired = os.path.abspath(paired)
            if not os.path.exists(paired):
                print(f"  WARNING: paired MSA for chain {chain} not found: {paired}")

        templates = []
        for tmpl in entry.get("templates", []) or []:
            tmpl = dict(tmpl)  # copy so we don't mutate the input
            mmcif_path = tmpl.get("mmcifPath", "")
            if mmcif_path:
                tmpl["mmcifPath"] = os.path.abspath(mmcif_path)
                if not os.path.exists(tmpl["mmcifPath"]):
                    print(f"  WARNING: template mmCIF for chain {chain} not found: {tmpl['mmcifPath']}")
            templates.append(tmpl)

        msa_map[chain] = {"unpaired": unpaired, "paired": paired, "templates": templates}
    return msa_map


def build_af3_json(job_name, chain_seqs, msa_map):
    """msa_map: {chain_id: {"unpaired": path, "paired": path, "templates": [...]}}"""
    sequences = []
    for chain_id, seq in chain_seqs.items():
        chain_msas = msa_map.get(chain_id, {})
        unpaired = chain_msas.get("unpaired", "")
        paired   = chain_msas.get("paired",   "")
        templates = chain_msas.get("templates", []) or []
        if not unpaired:
            print(f"  WARNING: no unpaired MSA specified for chain {chain_id}")
        if not paired:
            print(f"  WARNING: no paired MSA specified for chain {chain_id}")
        sequences.append({
            "protein": {
                "id": [chain_id],
                "sequence": seq,
                "unpairedMsaPath": unpaired,
                "pairedMsaPath":   paired,
                "unpairedMsa": "",
                "pairedMsa":   "",
                "templates":   templates,
            }
        })
    return {
        "name": job_name,
        "sequences": sequences,
        "modelSeeds": [1],
        "dialect": "alphafold3",
        "version": 1,
    }


def find_af3_paths():
    candidates = [
        ("/orcd/pool/005/keating_shared/alphafold3/alphafold3.sif",
         "/orcd/pool/005/keating_shared/alphafold3/alphafold3_data",
         "/orcd/pool/005/keating_shared/alphafold3/alphafold3_weights",
         "/orcd/pool/005/keating_shared/alphafold3/alphafold3"),
        ("/mnt/shared/shared_data/alphafold/alphafold3.sif",
         "/mnt/shared/shared_data/alphafold/alphafold3_data",
         "/mnt/shared/shared_data/alphafold/alphafold3_weights",
         "/mnt/shared/shared_data/alphafold/alphafold3"),
    ]
    for sif, dbs, weights, af_dir in candidates:
        if os.path.exists(sif):
            return sif, dbs, weights, af_dir
    raise FileNotFoundError("Could not locate alphafold3.sif")


def msa_bind_dirs(msa_map):
    """Return the set of unique parent directories holding MSA and template files."""
    dirs = set()
    for chain_msas in msa_map.values():
        for path in (chain_msas.get("unpaired", ""), chain_msas.get("paired", "")):
            if path:
                dirs.add(os.path.dirname(os.path.abspath(path)))
        for tmpl in chain_msas.get("templates", []) or []:
            mmcif_path = tmpl.get("mmcifPath", "")
            if mmcif_path:
                dirs.add(os.path.dirname(os.path.abspath(mmcif_path)))
    return dirs


def run_alphafold3(input_dir, out_dir, msa_map):
    sif, dbs, weights, af_dir = find_af3_paths()
    cmd = [
        "singularity", "exec", "--nv",
        "--pwd", "/root",
        "--bind", f"{input_dir}:/root/af_input",
        "--bind", f"{out_dir}:/root/af_output",
        "--bind", f"{weights}:/root/models",
        "--bind", f"{dbs}:/root/public_databases",
        "--bind", f"{af_dir}:/root/af",
    ]
    for msa_dir in msa_bind_dirs(msa_map):
        cmd += ["--bind", f"{msa_dir}:{msa_dir}"]
    cmd += [
        sif,
        "python", "/root/af/run_alphafold.py",
        "--model_dir=/root/models",
        "--input_dir=/root/af_input",
        "--db_dir=/root/public_databases",
        "--output_dir=/root/af_output",
        "--num_diffusion_samples=3",
    ]

    # Disable JAX GPU memory preallocation so concurrent AF3 processes can share one GPU.
    # Without this, each process tries to grab ~90% of GPU memory on first use and the
    # 2nd–Nth crash with "cuda_driver.cc: Check failed: stream != nullptr".
    env = os.environ.copy()
    env["SINGULARITYENV_XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    env["SINGULARITYENV_XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"
    subprocess.run(cmd, check=True, env=env)


def find_job_dirs(out_dir, job_name):
    """
    Return all output directories for a job, handling AF3's timestamped naming.
    AF3 creates one directory per seed: {job_lower} or {job_lower}_YYYYMMDD_HHMMSS.
    """
    job_lower = job_name.lower()
    matches = glob(os.path.join(out_dir, job_lower)) + \
              glob(os.path.join(out_dir, f"{job_lower}_[0-9]*"))
    return sorted(matches)


def af3_outputs_complete(out_dir, job_name):
    """
    With a single seed and --num_diffusion_samples=3, AF3 writes all 3 samples
    into one directory (possibly timestamped). Check for 3 summary files in any
    matching directory.
    """
    for job_dir in find_job_dirs(out_dir, job_name):
        summaries = glob(os.path.join(job_dir, "seed-*_sample-*", "*_summary_confidences.json"))
        if len(summaries) >= 3:
            return True
    return False


def find_seed_outputs(out_dir, job_name):
    """
    Return (pae_json, model_cif) pairs from the job directory that has all 3 samples.
    """
    for job_dir in sorted(find_job_dirs(out_dir, job_name), key=os.path.getmtime, reverse=True):
        pairs = []
        for seed_dir in sorted(glob(os.path.join(job_dir, "seed-*_sample-*"))):
            pae_files = [f for f in glob(os.path.join(seed_dir, "*confidences.json"))
                         if "summary" not in os.path.basename(f)]
            cif_files = glob(os.path.join(seed_dir, "*_model.cif"))
            if pae_files and cif_files:
                pairs.append((pae_files[0], cif_files[0]))
        if len(pairs) >= 3:
            return pairs
    return []


def run_ipsae(ipsae_script, pae_file, cif_file, pae_cutoff, dist_cutoff):
    subprocess.run(
        ["python", str(ipsae_script), str(pae_file), str(cif_file),
         str(pae_cutoff), str(dist_cutoff)],
        check=True,
    )
    pae_s = f"{pae_cutoff:02d}"
    dist_s = f"{dist_cutoff:02d}"
    stem = str(cif_file).replace(".cif", "")
    return Path(f"{stem}_{pae_s}_{dist_s}.txt")


def min_asym_ipsae(ipsae_txt):
    df = pd.read_csv(ipsae_txt, sep=r"\s+")
    asym = df[df["Type"] == "asym"]
    if asym.empty:
        return float("nan")
    return float(asym["ipSAE"].min())


def make_pisa_cfg(template_path, dest_path, session_prefix):
    """Copy template cfg, override SESSION_PREFIX so each row uses an isolated namespace."""
    with open(template_path) as f:
        lines = f.readlines()
    out_lines = []
    i = 0
    overridden = False
    while i < len(lines):
        out_lines.append(lines[i])
        stripped = lines[i].strip()
        if stripped == "SESSION_PREFIX" and i + 1 < len(lines):
            out_lines.append(f"{session_prefix}\n")
            i += 2
            overridden = True
            continue
        i += 1
    if not overridden:
        out_lines.append("SESSION_PREFIX\n")
        out_lines.append(f"{session_prefix}\n")
    with open(dest_path, "w") as f:
        f.writelines(out_lines)


def _find_first(parent, *tag_candidates):
    """Return first child element matching any of the given tag names (case variants)."""
    for tag in tag_candidates:
        el = parent.find(tag)
        if el is not None:
            return el
    return None


def _float_text(el):
    if el is None or el.text is None:
        return float("nan")
    try:
        return float(el.text)
    except ValueError:
        return float("nan")


def parse_pisa_interfaces_xml(xml_path):
    """Return (int_area, int_solv_en) for the largest interface, or (nan, nan)."""
    try:
        tree = ET.parse(xml_path)
    except (ET.ParseError, FileNotFoundError) as e:
        print(f"  PISA interfaces XML parse failed: {e}")
        return float("nan"), float("nan")
    root = tree.getroot()
    interfaces = root.findall(".//INTERFACE") or root.findall(".//interface")
    if not interfaces:
        return float("nan"), float("nan")

    def area_of(iface):
        return _float_text(_find_first(iface, "int_area", "INT_AREA"))

    best = max(interfaces, key=lambda i: (area_of(i) if not math.isnan(area_of(i)) else -1))
    return area_of(best), _float_text(_find_first(best, "int_solv_en", "INT_SOLV_EN"))


def parse_pisa_assemblies_xml(xml_path):
    """
    Return ΔG_diss for the most-stable multi-chain assembly, or NaN.
    Picks the assembly with largest size>=2; ties broken by highest dG_diss.
    """
    try:
        tree = ET.parse(xml_path)
    except (ET.ParseError, FileNotFoundError) as e:
        print(f"  PISA assemblies XML parse failed: {e}")
        return float("nan")
    root = tree.getroot()
    assemblies = root.findall(".//assembly") or root.findall(".//ASSEMBLY")
    if not assemblies:
        return float("nan")

    def size_of(a):
        s = _find_first(a, "size", "SIZE", "mmsize", "MMSIZE")
        try:
            return int(s.text) if s is not None and s.text else 0
        except ValueError:
            return 0

    def dg_of(a):
        return _float_text(_find_first(a, "diss_energy", "DISS_ENERGY", "dG_diss", "DG_DISS"))

    multi_chain = [a for a in assemblies if size_of(a) >= 2]
    if not multi_chain:
        return float("nan")
    best = max(multi_chain, key=lambda a: (size_of(a), (dg_of(a) if not math.isnan(dg_of(a)) else -1e9)))
    return dg_of(best)


def run_pisa(pisa_exe, cif_file, session_id, cfg_path):
    """
    Run PISA on a structure file (PDB or mmCIF) and return a dict:
        {"int_area": float, "int_solv_en": float, "dG_diss": float}
    NaN values indicate the corresponding quantity could not be parsed.

    int_solv_en is the solvation free energy of the largest interface (kcal/mol);
    dG_diss is PISA's free energy of dissociation for the largest multi-chain assembly,
    incorporating the entropy penalty: ΔG_diss = -ΔG_int - TΔS (Krissinel & Henrick 2007).
    """
    cif_file = str(cif_file)
    stem, _ext = os.path.splitext(cif_file)
    interfaces_xml = f"{stem}_pisa.xml"
    assemblies_xml = f"{stem}_pisa_assemblies.xml"

    nan_result = {"int_area": float("nan"), "int_solv_en": float("nan"), "dG_diss": float("nan")}

    try:
        subprocess.run(
            [pisa_exe, session_id, "-analyse", cif_file, "--as-is", str(cfg_path)],
            check=True,
        )
        with open(interfaces_xml, "w") as out:
            subprocess.run(
                [pisa_exe, session_id, "-xml", "interfaces", str(cfg_path)],
                check=True, stdout=out,
            )
        with open(assemblies_xml, "w") as out:
            subprocess.run(
                [pisa_exe, session_id, "-xml", "assemblies", str(cfg_path)],
                check=True, stdout=out,
            )
    except subprocess.CalledProcessError as e:
        print(f"  PISA failed: {e}")
        return nan_result
    finally:
        subprocess.run([pisa_exe, session_id, "-erase", str(cfg_path)], check=False)

    int_area, int_solv_en = parse_pisa_interfaces_xml(interfaces_xml)
    dG_diss = parse_pisa_assemblies_xml(assemblies_xml)
    return {"int_area": int_area, "int_solv_en": int_solv_en, "dG_diss": dG_diss}


def safe_token(s):
    return s.replace(":", "_").replace(",", "_").replace(" ", "")


# Module-level state populated in worker processes by _init_worker
_WORKER_SHARED = None
_WORKER_INPUT_DIR = None


def _init_worker(shared, counter, lock):
    """Pool initializer: assign each worker a unique input_dir to avoid AF3 collisions."""
    global _WORKER_SHARED, _WORKER_INPUT_DIR
    _WORKER_SHARED = shared
    with lock:
        wid = counter.value
        counter.value += 1
    _WORKER_INPUT_DIR = os.path.join(shared["out_dir"], f"inputs_worker_{wid}")
    os.makedirs(_WORKER_INPUT_DIR, exist_ok=True)


def _process_row_worker(work_item):
    """Wrapper for Pool: pulls shared state from module globals."""
    return process_row(work_item, _WORKER_SHARED, _WORKER_INPUT_DIR)


_NAN_PISA = {"int_area": float("nan"), "int_solv_en": float("nan"), "dG_diss": float("nan")}


def process_row(work_item, shared, input_dir):
    """
    Run the AF3 -> ipSAE -> PISA pipeline for one CSV row.
    Returns (row_idx, ipsae_value, pisa_result_dict).
    """
    i = work_item["row_idx"]
    mut_string = work_item["mut_string"]
    mut_seqs = work_item["mut_seqs"]
    tokens = work_item["tokens"]
    n_total = work_item["n_total"]

    base_name      = shared["base_name"]
    out_dir        = shared["out_dir"]
    msa_map        = shared["msa_map"]
    pisa_exe       = shared["pisa_exe"]
    pisa_cfg       = shared["pisa_cfg"]
    pisa_cfg_dir   = shared["pisa_cfg_dir"]
    pisa_name_base = shared["pisa_name"]
    ipsae_script   = shared["ipsae_script"]
    pae_cutoff     = shared["pae_cutoff"]
    dist_cutoff    = shared["dist_cutoff"]

    print(f"\n=== [{i+1}/{n_total}] {mut_string} ===", flush=True)
    print(f"  Applied {len(tokens)} mutation(s): {tokens}", flush=True)

    job_name = safe_token(f"{base_name}__{'_'.join(tokens)}")
    json_path = os.path.join(input_dir, f"{job_name}.json")
    failed_marker = os.path.join(out_dir, f"{job_name}.failed")

    if os.path.exists(failed_marker):
        print(f"  WARNING: {job_name} previously failed (marker exists). Skipping permanently.", flush=True)
        return i, float("nan"), dict(_NAN_PISA)

    if not af3_outputs_complete(out_dir, job_name):
        with open(json_path, "w") as f:
            json.dump(build_af3_json(job_name, mut_seqs, msa_map), f, indent=2)
        af3_error = None
        try:
            run_alphafold3(input_dir, out_dir, msa_map)
        except subprocess.CalledProcessError as e:
            af3_error = e
        try:
            os.remove(json_path)
        except OSError:
            pass

        # Mark failure if AF3 raised OR if AF3 returned success but outputs are
        # incomplete (silent crashes, e.g. CUDA OOM mid-sample). The marker
        # prevents future runs from spawning duplicate timestamped directories.
        if af3_error is not None or not af3_outputs_complete(out_dir, job_name):
            reason = f"AF3 raised: {af3_error}" if af3_error is not None \
                     else "AF3 exited 0 but output is incomplete (likely partial crash)"
            print(f"  WARNING: AF3 failed for {job_name}: {reason}", flush=True)
            print(f"  Writing failure marker: {failed_marker}", flush=True)
            open(failed_marker, "w").close()
            return i, float("nan"), dict(_NAN_PISA)
    else:
        print(f"  AF3 outputs already complete for {job_name}, skipping inference.", flush=True)

    seed_outputs = find_seed_outputs(out_dir, job_name)
    if not seed_outputs:
        print(f"  No AF3 outputs found for {job_name}", flush=True)
        return i, float("nan"), dict(_NAN_PISA)

    best_min = -math.inf
    best_cif = None
    for pae_file, cif_file in seed_outputs:
        try:
            ipsae_txt = run_ipsae(ipsae_script, pae_file, cif_file, pae_cutoff, dist_cutoff)
            m = min_asym_ipsae(ipsae_txt)
        except (subprocess.CalledProcessError, FileNotFoundError, ValueError) as e:
            print(f"  ipSAE failed for {cif_file}: {e}", flush=True)
            continue
        print(f"  {os.path.basename(os.path.dirname(cif_file))}: min_asym_ipSAE = {m:.4f}", flush=True)
        if not math.isnan(m) and m > best_min:
            best_min = m
            best_cif = cif_file

    if best_cif is None:
        return i, float("nan"), dict(_NAN_PISA)

    print(f"  Best model: {best_cif} (ipSAE={best_min:.4f})", flush=True)
    # Use job_name in the PISA session id to keep sessions globally unique across workers
    session_name = f"{pisa_name_base}_{job_name}"
    row_cfg = os.path.join(pisa_cfg_dir, f"{session_name}.cfg")
    make_pisa_cfg(pisa_cfg, row_cfg, session_prefix=f"{session_name}_")
    pisa_result = run_pisa(pisa_exe, best_cif, session_name, row_cfg)
    print(f"  int_area={pisa_result['int_area']:.2f}  "
          f"int_solv_en={pisa_result['int_solv_en']:.4f}  "
          f"dG_diss={pisa_result['dG_diss']:.4f}", flush=True)

    return i, best_min, pisa_result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mutations_csv", required=True)
    ap.add_argument("--fasta", required=True)
    ap.add_argument("--msa_json", required=True,
                    help='JSON file mapping chain -> {"unpaired": path, "paired": path}.')
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--pisa_exe", required=True)
    ap.add_argument("--pisa_cfg", required=True,
                    help="Path to PISA config template (will be copied per row).")
    ap.add_argument("--pisa_name", required=True,
                    help="Base session name for PISA; row index is appended.")
    ap.add_argument("--ipsae_script", required=True)
    ap.add_argument("--pae_cutoff", type=int, default=10)
    ap.add_argument("--dist_cutoff", type=int, default=15)
    ap.add_argument("--max_parallel", type=int, default=1,
                    help="Number of concurrent worker processes (one AF3 call per worker). "
                         "Each worker uses its own input_dir; all share one GPU. "
                         "Tune this so that N * (per-AF3 GPU memory) fits on the H100.")
    args = ap.parse_args()

    # Fail fast if any required input path is missing — better than discovering
    # mid-run after AF3 has already created output directories.
    for label, p in [("--fasta", args.fasta),
                     ("--msa_json", args.msa_json),
                     ("--pisa_exe", args.pisa_exe),
                     ("--pisa_cfg", args.pisa_cfg),
                     ("--ipsae_script", args.ipsae_script),
                     ("--mutations_csv", args.mutations_csv)]:
        if not os.path.exists(p):
            sys.exit(f"ERROR: {label} path does not exist: {p}")

    out_dir = os.path.abspath(args.out_dir)
    input_dir = os.path.join(out_dir, "inputs")
    pisa_cfg_dir = os.path.join(out_dir, "pisa_cfgs")
    os.makedirs(input_dir, exist_ok=True)
    os.makedirs(pisa_cfg_dir, exist_ok=True)

    msa_map = load_msa_map(args.msa_json)
    ipsae_script = Path(args.ipsae_script).resolve()

    df = pd.read_csv(args.mutations_csv)
    if "mutations" not in df.columns:
        raise ValueError(f"Expected 'mutations' column; got {list(df.columns)}")

    # Sanity-check CSV parsing: if stability_score is non-numeric it means a
    # mutation pair was unquoted and pandas split it across columns.
    if "stability_score" in df.columns:
        bad_rows = df[pd.to_numeric(df["stability_score"], errors="coerce").isna()]
        if not bad_rows.empty:
            raise ValueError(
                f"CSV appears mis-parsed: {len(bad_rows)} row(s) have non-numeric "
                f"stability_score (e.g. row {bad_rows.index[0]}: "
                f"stability_score={bad_rows.iloc[0]['stability_score']!r}). "
                f"Ensure every mutations field in the CSV is quoted, e.g. "
                f'"B:W102E,B:N89H",-0.035,...'
            )

    print(f"Loaded {len(df)} rows. First mutations value: {df['mutations'].iloc[0]!r}")

    base_name, base_chain_seqs = parse_fasta(args.fasta)

    # Prepend wildtype row with zero scores
    score_cols = [c for c in ["stability_score", "binding_score", "uniqueness_score"] if c in df.columns]
    wt_row = {"mutations": "WT", **{c: 0.0 for c in score_cols}}
    df = pd.concat([pd.DataFrame([wt_row]), df], ignore_index=True)

    # Build work items: pre-resolve sequences and tokens for each row.
    # Rows that fail to parse get fixed NaN results and are not dispatched.
    n_total = len(df)
    ipsae_vals     = [float("nan")] * n_total
    int_area_vals  = [float("nan")] * n_total
    int_solv_vals  = [float("nan")] * n_total
    dG_diss_vals   = [float("nan")] * n_total
    work_items = []
    for i, row in df.iterrows():
        mut_string = str(row["mutations"])
        if mut_string == "WT":
            mut_seqs, tokens = base_chain_seqs, ["WT"]
        else:
            try:
                mut_seqs, tokens = apply_mutations(base_chain_seqs, mut_string)
            except ValueError as e:
                print(f"[row {i+1}] Mutation parse/WT error: {e}")
                continue
        work_items.append({
            "row_idx": i,
            "mut_string": mut_string,
            "mut_seqs": mut_seqs,
            "tokens": tokens,
            "n_total": n_total,
        })

    shared = {
        "base_name": base_name,
        "out_dir": out_dir,
        "msa_map": msa_map,
        "pisa_exe": args.pisa_exe,
        "pisa_cfg": args.pisa_cfg,
        "pisa_cfg_dir": pisa_cfg_dir,
        "pisa_name": args.pisa_name,
        "ipsae_script": str(ipsae_script),
        "pae_cutoff": args.pae_cutoff,
        "dist_cutoff": args.dist_cutoff,
    }

    def _store(i, ipsae, pisa):
        ipsae_vals[i]    = ipsae
        int_area_vals[i] = pisa["int_area"]
        int_solv_vals[i] = pisa["int_solv_en"]
        dG_diss_vals[i]  = pisa["dG_diss"]

    if args.max_parallel == 1:
        # Sequential path: keeps existing behavior, single inputs/ dir
        for w in work_items:
            i, ipsae, pisa = process_row(w, shared, input_dir)
            _store(i, ipsae, pisa)
    else:
        print(f"Running with {args.max_parallel} parallel workers on the same GPU.")
        ctx = mp.get_context("spawn")
        counter = ctx.Value("i", 0)
        lock = ctx.Lock()
        with ctx.Pool(args.max_parallel,
                      initializer=_init_worker,
                      initargs=(shared, counter, lock)) as pool:
            for i, ipsae, pisa in pool.imap_unordered(_process_row_worker, work_items):
                _store(i, ipsae, pisa)

    df["ipSAE"]      = ipsae_vals
    df["int_area"]   = int_area_vals
    df["dG_binding"] = int_solv_vals  # legacy column = int_solv_en (kept for backward compat)
    df["dG_diss"]    = dG_diss_vals   # primary metric per Krissinel & Henrick 2007

    out_csv = os.path.join(
        out_dir,
        f"{Path(args.mutations_csv).stem}_with_af3.csv"
    )
    df.to_csv(out_csv, index=False)
    print(f"\nWrote {out_csv}")


if __name__ == "__main__":
    main()
