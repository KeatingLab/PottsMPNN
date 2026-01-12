# PottsMPNN

Code for running PottsMPNN to generate protein sequences and predict energies of mutations.

---

## 1. Running PottsMPNN

There are three ways to run the model, ranging from easiest (cloud-based) to advanced (command-line).

### Level 1: Google Colab (Easiest)

Run the model entirely in your browser using Google's free GPU resources. No local installation is required.

* **Sequence Generation:** [Use this file to design new sequences for a backbone](https://colab.research.google.com/drive/1Jx447uZHwi_pvLbzYtdL961vsatAjlnd?usp=sharing)
* **Energy Prediction:** [Use this file to predict ΔΔG values for specific mutations.](https://colab.research.google.com/drive/1nAWcQXW_GQkyyN0X2s0G68w-8y0wDbpx?usp=sharing)

### Level 2: Local Jupyter Notebooks

If you have set up the installation environment (see below), you can run the interactive notebooks locally. This allows for easier file management and faster execution on local GPUs.

1. Start Jupyter:
```bash
jupyter notebook

```


2. Open `sample_seqs.ipynb` for sequence generation.
3. Open `energy_prediction.ipynb` for mutational scoring.

### Level 3: Command Line Interface

For batch processing or integration into pipelines, run the Python scripts directly using a YAML configuration file. Note that the configuration files by default set `dev` to `cuda`. If your system does not have a PyTorch installation with CUDA, you must set `dev` to `cpu`.


**Generating Sequences:**

```bash
python sample_seqs.py --config inputs/example_config_sample_seqs.yaml

```

**Predicting Energies:**

```bash
python energy_prediction.py --config inputs/example_config_energy_prediction.yaml

```

You can verify the code is working by comparing the outputs to the files in the `\outputs` directory in this repository. Because the sampling temperature affects the sequence generation file, to validate sequence generation update the YAML file to have a very low temperature and validate against `\outputs\example_sequence_outputs\low_temp_cuda.fasta` or `\outputs\example_sequence_outputs\low_temp_cpu.fasta`.

---

## 2. Installation

We recommend using **Conda** to manage the environment and **pip** to install dependencies.'

### Step 0: Decide If Local Installation Is Needed
If you are interested in trying out PottsMPNN, you can use the Google Colabs (see above) without needing to install anything.

### Step 1: Create a Conda Environment
Create a clean environment with Python 3.10:
```bash
conda create -n PottsMPNN python=3.10
conda activate PottsMPNN

```

### Step 2: Install PyTorch

Install the version of PyTorch compatible with your system (CUDA/GPU recommended). Visit the [official PyTorch installation page](https://pytorch.org/get-started/locally/) to get the correct command. For example:

**Linux with CUDA 12.1:**

```bash
pip install torch torchvision torchaudio --index-url [https://download.pytorch.org/whl/cu124](https://download.pytorch.org/whl/cu124)

```

**Mac (CPU/MPS):**

```bash
pip install torch torchvision torchaudio

```

### Step 3: Install Dependencies

Install the required Python packages using the provided requirements file:

```bash
pip install -r requirements.txt

```

---

## 3. Configuration Options

Both pipelines use a configuration dictionary (or YAML file) to control the model. Example configurations can be found in the `inputs/` directory.

### Model weights

We provide vanilla and soluble model weights. For each, we provide three versions of PottsMPNN: one trained with 0.2 Å of noise and MSAs (which we recommend as the default); one trained with 0.3 Å of noise (which performs the best if you do not want to use a model trained with MSAs); and one trained with 0.2 Å of noise (which provides a direct comparison to the default version of ProteinMPNN).

### Sequence Generation Options (`sample_seqs`)

### Data input

* **`input_dir`**: Path to directory containing structures in `.pdb` format.
* **`input_list`**: Path to a `.txt` file identifying structures for which to sample sequences. Chain information (i.e., which chains to design and which are visible to the model) can be specified using the `'|'` token to distinguish between groups of designed and visible chains, which in a group should be split using the `':'` token. For example, a line in the file of `prot|A:B:C|D:E` indicates that chains A, B, and C should be designed and chains D and E should be visible for structure prot.pdb. You can put the same structure on multiple lines so long as there are different combinations of designed and visible chains. If no chain information is specified, all chains will be designed.
* **`chain_dict_json`**: Path to JSON identifying which chains should be designed and which are visible. For example, an entry in the JSON of `prot: [[A, B, C], [D, E]]` indicates that chains A, B, and C should be designed and chains D and E should be visible for structure prot.pdb. Providing a JSON file will overwrite information in **`input_list`**.

#### Sampling & Optimization

* **`inference.num_samples`**: (int) Number of sequences to generate per structure. Must be `1` if running optimization.
* **`inference.temperature`**: (float) Sampling temperature. Lower values (e.g., 0.1) produce more probable sequences; higher values (e.g., 1.0) add diversity.
* **`inference.noise`**: (float) Amont of noise to add to structures during inference (model only evaluated with 0 noise).
* **`inference.skip_gaps`**: (bool) Whether to skip gaps in structure (default False, but for some downstream applications like forward folding it can be convenient to set to True).
* **`inference.fix_decoding_order`**: (bool) Whether to use a fixed decoding order for each structure.
* **`inference.decoding_order_offset`**: (bool) Offset if you want a fixed decoding order different from the standard fixed decoding order.
* **`inference.optimization_mode`**: Optimization protocol to use:
* `"none"`: No optimization (just autoregressive sampling).
* `"potts"`: Optimizes sequence using Potts energy.
* `"nodes"`: Optimizes node features.
* **`inference.binding_energy_optimization`**: How to optimize using binding energies:
* `"none"`: Optimize stability only.
* `"both"`: Jointly optimize stability and binding affinity.
* `"only"`: Optimize binding affinity only.
* **`inference.binding_energy_json`**: Path to JSON with information about how chains should be separated for binding energy calculation for optimization (required for binding energy optimization). For example, `prot: [[A, B], [C, D]]` indicates that the binding energy should be calculated between the A-B and C-D complexes; `prot: [[A, B], [C], [D]]` indicates that the binding energy should be calculated between the A-B complex and chains C and D.
* **`inference.binding_energy_cutoff`**: (float) Angstrom cutoff for which residues to optimize with binding energies (binding energies only accurate for residues close to the interface).
* **`inference.optimize_pdb`**: (bool) Optimize sequences found in the input `.pdb` files.
* **`inference.optimize_fasta`**: (bool) Optimize sequences found in an input `.fasta` file.
* **`inference.write_pdb`**: (bool) Write new `.pdb` files with the best sequences found for each structure.

#### Constraints & Biases (inhereted from ProteinMPNN)
* **`inference.fixed_positions_json`**: Path to JSON defining 1-indexed positions to fix (keep as wildtype).
* **`inference.pssm_json`**: Path to JSON containing Position-Specific Scoring Matrix (bias per position).
* **`inference.omit_AA_json`**: Path to JSON defining amino acids to ban at specific positions.
* **`inference.bias_AA_json`**: Path to JSON defining global amino acid biases.
* **`inference.omit_AAs`**: List of amino acids to globally omit from design (e.g., `['C', 'W']`).
* **`inference.tied_positions_json`**: Path to JSON containing tied position information for each structure.
* **`inference.bias_by_res_json`**: Path to JSON containing residue bias info (shape chain length by vocab) for each chain in each structure.
* **`inference.pssm_threshold`**: (float) A value between `-inf` and `+inf` to restrict per position AAs.
* **`inference.pssm_multi`**: (float) A value between `[0.0, 1.0]`, `0.0` means do not use pssm, `1.0` ignore MPNN predictions.
* **`inference.pssm_log_odds_flag`**: (bool) `0` for False, `1` for True.
* **`inference.pssm_bias_flag`**: (bool) `0` for False, `1` for True.

### Energy Prediction Options (`energy_prediction`)

### Data input

* **`input_dir`**: Path to directory containing structures in .pdb format.
* **`input_list`**: Path to a `.txt` file containing the list of structures to process.
* **`mutant_fasta`**: Path to FASTA file with information on which mutations to predict (if `None` and `mutant_csv` is `None`, do deep mutational scan by mutating every position to every possible canonical amino acid). Header information in FASTA should be of the format `prot|<mutant_chains>` or `prot|<mutant_chains>|<experimental_energy>` where `<mutant_chains>` is a list of chain identifiers seperated by the `|` token indicating which chains in the protein have a mutation. If experimental energies are provided, they will be compared to the predicted energies. The sequence lines in the FASTA must only include sequences for the chains indicated in the header. See `inputs/example_energy_input.fasta` for an example.
* **`mutant_csv`**: Path to CSV file with information on which mutations to predict (if `None` and `mutant_fasta` is `None`, do deep mutational scan by mutating every position to every possible canonical amino acid). The CSV must have columns `pdb`, `mut_type`, and `chain`. `pdb` indicates the protein. `mut_type` is of format `<wt><pos><mut>` where `<wt>` is the wild-type residue, `<pos>` is a 0-indexed position in the chain as listed in the corresponding `.pdb` file, and `<mut>` is the mutant residue. If there are multiple mutations per sequence, concatenate them the `':'` token. `chain` is a list of chain identifiers for the mutated positions, concatenated with the `':'` token. The CSV can optionally have the `ddG_expt` column, which if provided will be compared against the predicted energies.

### Prediction options
* **`inference.ddG`**: (bool) If True, computes ΔΔG (mutant - wildtype); if False, outputs raw ΔG (True by default, ΔG values are less interpretable).
* **`inference.mean_norm`**: (bool) Whether to center ΔΔG predictions around 0 (can sometimes help compare predictions from one protein to another).
* * **`inference.max_tokens`**: (int) Max tokens to use when batching energy predictions (lower is slower but less memory intensive).
* **`inference.filter`**: (bool) Whether to only return predictions for mutant sequences with experimental energies.
* **`inference.binding_energy_json`**: Path to JSON with information about how chains should be separated for binding energy calculation for optimization (required for binding energy optimization). For example, `prot: [[A, B], [C, D]]` indicates that the binding energy should be calculated between the A-B and C-D complexes; `prot: [[A, B], [C], [D]]` indicates that the binding energy should be calculated between the A-B complex and chains C and D.
* **`inference.noise`**: (float) Amont of noise to add to structures during inference (model only evaluated with 0 noise).
* **`inference.skip_gaps`**: (bool) Whether to skip gaps in structure (default False).
* **`inference.chain_ranges`**: Path to JSON specifying ranges of positions (1-indexed) for heatmap plotting. For example, `prot: {A: [1,50], B: [20:30]}` indicates to plot residues 1 to 50 from chain A and residues 20 to 30 from chain B.

---

## 4. PottsMPNN as a Replacement for ProteinMPNN

PottsMPNN was built from the ProteinMPNN architecture and can accomplish any task ProteinMPNN can do. The scripts provided above take advantage of the Potts model learned by PottsMPNN, but you can also use PottsMPNN as a direct ProteinMPNN replacement using the model weights in `\proteinmpnn_compatible_model_weights`.

## 5. Extracting Energy Tables

To extract Potts model energy tables for an input `.pdb` file `input_pdb` and list of chains `partition`, you can use the following code:

```
from potts_mpnn_utils import PottsMPNN, parse_PDB
from run_utils import get_etab
from etab_utils import expand_etab

# Load model checkpoint and construct the PottsMPNN model for inference
checkpoint = torch.load(cfg.model.check_path, map_location='cpu', weights_only=False) 
model = PottsMPNN(ca_only=False, num_letters=cfg.model.vocab, vocab=cfg.model.vocab, node_features=cfg.model.hidden_dim, edge_features=cfg.model.hidden_dim, hidden_dim=cfg.model.hidden_dim, 
                        potts_dim=cfg.model.potts_dim, num_encoder_layers=cfg.model.num_layers, num_decoder_layers=cfg.model.num_layers, k_neighbors=cfg.model.num_edges, augment_eps=cfg.inference.noise)
model.load_state_dict(checkpoint['model_state_dict'], strict=False)
model.eval()
model = model.to(dev)
for param in model.parameters():
    param.requires_grad = False

pdb_data = parse_PDB(input_pdb, skip_gaps=cfg.inference.skip_gaps)
etab, E_idx, wt_seq = get_etab(model, pdb_data, cfg, partition)
sparse_etab = expand_etab(etab, E_idx)
```
`etab` is the dense energy table in shape `[1, L, k, h, h]` where `L` is the sequence length, `k=48`, and `h=22`. `sparse_etab` is the energy table in shape `[1, L, L, h, h]`. `E_idx` contains the neighbor indices for each residue and is of shape `[1, L, k]`.

---

## 6. Energy Benchmark Datasets

The energy datasets used in the paper are located at `\energy_benchmark_datasets`. For Megascale and FireProt, we removed proteins from the datasets that were present in the training set. See the following citations for each dataset:
* Megascale: K Tsuboyama, et al., Mega-scale experimental analysis of protein folding stability in biology
and design. Nature 620, 434–444 (2023). [https://doi.org/10.1038/s41586-023-06328-6](https://doi.org/10.1038/s41586-023-06328-6)
* FireProt: J Stourac, et al., FireProtDB: database of manually curated protein stability data. Nucleic
Acids Res. 49, D319–D324 (2021). [https://doi.org/10.1093/nar/gkaa981](https://doi.org/10.1093/nar/gkaa981)
* SARS-CoV-2: TN Starr, et al., Deep Mutational Scanning of SARS-CoV-2 Receptor Binding Domain Reveals
Constraints on Folding and ACE2 Binding. Cell 182, 1295–1310.e20 (2020). [https://doi.org/10.1016/j.cell.2020.08.012](https://doi.org/10.1016/j.cell.2020.08.012)
