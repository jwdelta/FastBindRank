# FastBindRank

FastBindRank is a command-line distillation workflow that uses affinity predictions from high-fidelity structure-based models to train a fast surrogate ranking model for compound library preparation, iteration-based sampling, model selection, full-library rescoring, and top-hit structural diversity analysis. Boltz-2 is currently used as the teacher model, given its near–free-energy perturbation (FEP) performance in protein–ligand binding affinity prediction. The framework is model-agnostic and can readily incorporate newer or more advanced teacher models as they become available.

<p align="center">
  <img src="https://github.com/user-attachments/assets/f7b920ae-4a2a-4681-8b4e-4528c4d49081" alt="Workflow" style="width:700px;">
</p>


## Workflow Overview

The current workflow is organized into these stages:

1. Prepare a PubChem-based screening library
2. Repeat iteration-level sampling, affinity prediction, merging, and model training
3. Summarize model runs and select the best model from the last iteration
4. Predict the full library with the selected best model
5. Select top hits, perform Boltz-2 rescoring, apply multi-criteria candidate filtering, and conduct structural diversity selection

## Software Requirements

FastBindRank currently depends on the following software being available in your environment:

| Software | Purpose |
| --- | --- |
| `python3` | Helper scripts for sampling, extraction, summarization, and postprocessing |
| `boltz` | Boltz-2 affinity prediction and top-hit rescoring |
| `RDKit` | Morgan fingerprint generation and molecular property annotation |
| `numpy` | Numeric array handling |
| `pandas` | Table processing for summary and annotation outputs |
| `matplotlib` | PCA figure generation |
| `scikit-learn` | PCA used in clustering visualization |
| `torch` | DL model training and inference |
| `scipy` | Correlation metrics used during model evaluation |

Depending on your environment, you may also need:

- GPU drivers and CUDA support for Boltz-2 and PyTorch GPU runs
- a job scheduler such as dSQ if you want to submit chunked affinity prediction as job arrays

## Main Commands

Before first use on a new machine or cluster copy, make sure the shell entry scripts are executable:

```bash
chmod +x bin/fastbindrank
chmod +x bin/commands/*.sh
```

### Common Parameters

| Parameter | Meaning |
| --- | --- |
| `--workspace` | Root working directory. This stores the prepared PubChem download and library files. |
| `--results-dir` | Directory for iteration outputs, model summaries, best-model bundles, and downstream analysis results. If omitted, the default is `<workspace>/results/`. |
| `--selection-dir` | Path to a specific best-model selection bundle under `<results-dir>/best_model/`. Use this when you want to operate on one explicit selected model instead of the latest one. |
| `--protein-name` | Short identifier used for Boltz-2 input and output naming. |
| `--protein-yaml-template` | Boltz-2 YAML template containing the protein definition(s), ligand block, and affinity property block. Use `__LIGAND_SMILES__` as the ligand SMILES placeholder; `__LIGAND_ID__` is also supported if you want to inject the compound ID. For custom MSA runs, include `msa: '__MSA_PATH__'` in the protein block and pass `--msa-path`. |

### 1. Prepare the library

```bash
bin/fastbindrank library prepare-pubchem \
  --workspace /path/to/workspace
```

This creates:

```text
<workspace>/
├── pubchem/
└── library/
    ├── smiles.txt
    ├── smiles_chunks/
    └── morgan_fingerprints/
```

The Morgan fingerprint directory contains chunk files named like `morgan_chunk_000.csv`, `morgan_chunk_001.csv`, and so on.

Key parameters:

| Parameter | Meaning |
| --- | --- |
| `--chunk-lines` | Number of compounds per split SMILES chunk in `library/smiles_chunks/`. Default: `1000000`. |
| `--workers` | Number of worker processes used when generating Morgan fingerprints. Default: `8`. |
| `--skip-download` | Reuse existing downloaded or decompressed PubChem files. |
| `--skip-split` | Reuse existing `smiles.txt` and `smiles_chunks/`. |
| `--skip-fingerprints` | Skip Morgan fingerprint generation. |
| `--cleanup-downloads` | Remove the `pubchem/` download directory after the library has been generated successfully. |

### 2. Iteration Workflow

```bash
bin/fastbindrank iterations prepare \
  --workspace /path/to/workspace \
  --results-dir /path/to/results/project_x \
  --iteration 1 \
  --exclude-ids-file /path/to/excluded_ids.txt
```

Important behavior:

- `iteration_1` samples the training split, tuning split, and test split
- `iteration > 1` samples only the training split
- `iteration > 1` reuses the tuning split and test split from `iteration_1`
- compounds already recorded in `<results-dir>/used_compound_ids.tsv` are automatically excluded from future sampling
- compounds listed in `--exclude-ids-file` are also excluded when you need an additional external blacklist

Key parameters for `iterations prepare`:

| Parameter | Meaning |
| --- | --- |
| `--iteration` | Iteration index to prepare, such as `1`, `2`, or `3`. |
| `--exclude-ids-file` | Optional extra blacklist of compound IDs that must never be sampled. |
| `--training-split` | Number of compounds to sample into the training split. Default: `250000`. |
| `--tuning-split` | Number of compounds to sample into the tuning split for `iteration_1`. Default: `100000`. |
| `--test-split` | Number of compounds to sample into the test split for `iteration_1`. Default: `250000`. |
| `--workers` | Number of worker processes used during sampling and extraction. Default: `10`. |
| `--seed` | Random seed used for reproducible sampling. Default: `42`. |

Example using automatic MSA generation:

```bash
bin/fastbindrank affinity predict-chunk \
  --workspace /path/to/workspace \
  --results-dir /path/to/results/project_x \
  --protein-name ExampleTarget \
  --protein-yaml-template examples/protein_templates/single_chain.yaml \
  --iteration 1 \
  --split training_split \
  --start-line 1 \
  --end-line 1000
```

Example using a precomputed MSA:

```bash
bin/fastbindrank affinity predict-chunk \
  --workspace /path/to/workspace \
  --results-dir /path/to/results/project_x \
  --protein-name ExampleTarget \
  --protein-yaml-template examples/protein_templates/single_chain_custom_msa.yaml \
  --msa-path /path/to/target.a3m \
  --iteration 1 \
  --split training_split \
  --start-line 1 \
  --end-line 1000
```

The protein YAML template lets you support a single chain, a homodimer such as `id: [A, B]`, or multiple separately defined chains. The template should already contain the full protein definition(s), plus a ligand entry whose SMILES value is `__LIGAND_SMILES__`.

By default, `affinity predict-chunk` runs Boltz-2 with `--use_msa_server`, so Boltz-2 generates MSA data automatically. If you already have a precomputed MSA, pass `--msa-path /path/to/msa.a3m`; FastBindRank will replace `__MSA_PATH__` in the YAML template and will not pass `--use_msa_server`[recommended].

Example templates are provided in:

- `examples/protein_templates/single_chain.yaml`
- `examples/protein_templates/homodimer.yaml`
- `examples/protein_templates/heterodimer.yaml`
- `examples/protein_templates/single_chain_custom_msa.yaml`

Example template using automatic MSA generation:

```yaml
version: 1
sequences:
  - protein:
      id: A
      sequence: YOUR_SEQUENCE
  - ligand:
      id: L
      smiles: '__LIGAND_SMILES__'
properties:
  - affinity:
      binder: L
```

Example template using a user-provided MSA:

```yaml
version: 1
sequences:
  - protein:
      id: A
      sequence: YOUR_SEQUENCE
      msa: '__MSA_PATH__'
  - ligand:
      id: L
      smiles: '__LIGAND_SMILES__'
properties:
  - affinity:
      binder: L
```

Key parameters for `affinity predict-chunk`:

| Parameter | Meaning |
| --- | --- |
| `--split` | Which split file to process, typically `training_split`, `tuning_split`, or `test_split`. |
| `--start-line` | First 1-based line number in the split file for this chunk. This parameter is intended for job-array style chunking, where each GPU job processes a fixed subset of compounds. |
| `--end-line` | Last 1-based line number in the split file for this chunk. This parameter is intended for job-array style chunking, where each GPU job processes a fixed subset of compounds. |
| `--boltz` | Boltz-2 executable name or path. |
| `--boltz-threads` | Preprocessing thread count passed to Boltz-2. Default: `1`. |
| `--accelerator` | Accelerator passed to Boltz-2, typically `gpu`. |
| `--msa-path` | Optional precomputed MSA file, such as an `.a3m`. When provided, the YAML template must contain `__MSA_PATH__`, and Boltz-2 is run without `--use_msa_server`. |

Strong recommendation: use `--start-line` and `--end-line` through an HPC job-array workflow so that each GPU job handles a fixed chunk of compounds from the split file.

After all chunks for one split finish, merge them:

```bash
bin/fastbindrank affinity merge-chunks \
  --workspace /path/to/workspace \
  --results-dir /path/to/results/project_x \
  --protein-name ExampleTarget \
  --iteration 1 \
  --split training_split
```

```bash
bin/fastbindrank iterations train \
  --workspace /path/to/workspace \
  --results-dir /path/to/results/project_x \
  --iteration 1
```

Training behavior:

- runs the full predefined hyperparameter search space automatically
- keeps `hard_label=false`
- uses the requested iteration for the training split
- uses `iteration_1` for the tuning split and test split
- finetunes from the previous iteration checkpoint when a matching configuration exists

Key parameters for `iterations train`:

| Parameter | Meaning |
| --- | --- |
| `--iteration` | Iteration index to train. |
| `--epochs` | Maximum number of epochs for each hyperparameter configuration. Default: `1000`. |
| `--batch-size` | Batch size used during model training. Default: `512`. |
| `--train-workers` | PyTorch data loader worker count. Default: `1`. |
| `--early-stop-patience` | Number of unimproved epochs before early stopping. Default: `3`. |

### 3. Summarize model runs and select the best model

```bash
bin/fastbindrank results summarize-models \
  --workspace /path/to/workspace \
  --results-dir /path/to/results/project_x
```

This writes:

```text
<results-dir>/
├── model_selection/
│   └── soft_label_model_summary.tsv
└── best_model/
    └── iteration_<N>_best_tuning_soft_pr_auc/
```

Key parameters:

| Parameter | Meaning |
| --- | --- |
| `--results-dir` | Result directory to scan for all iteration model outputs. |

### 4. Predict the full library with the selected best model

```bash
bin/fastbindrank results predict-all-best \
  --workspace /path/to/workspace \
  --results-dir /path/to/results/project_x
```

This writes:

```text
<selection-dir>/library_predictions/
├── chunks/
├── library_predictions.csv
└── library_predictions_ranked.csv
```

Key parameters:

| Parameter | Meaning |
| --- | --- |
| `--selection-dir` | Optional explicit best-model bundle to use for full-library prediction. |

### 5. Select and rescore top hits

Select top hits:

```bash
bin/fastbindrank top-hits select \
  --workspace /path/to/workspace \
  --results-dir /path/to/results/project_x \
  --top-k 1000
```

Key parameter:

| Parameter | Meaning |
| --- | --- |
| `--top-k` | Number of highest-ranked compounds to keep from the full-library prediction list. No default; this value must be specified by the user. |

Run chunked Boltz-2 rescoring:

```bash
bin/fastbindrank top-hits rescore-chunk \
  --workspace /path/to/workspace \
  --results-dir /path/to/results/project_x \
  --protein-name ExampleTarget \
  --protein-yaml-template /path/to/example_target.yaml \
  --start-line 1 \
  --end-line 200
```

This chunked rescoring step uses the selected top-hits table and is intended for job-array style execution, just like the earlier affinity prediction step.

Merge, annotate, filter, and cluster:

```bash
bin/fastbindrank top-hits cluster \
  --workspace /path/to/workspace \
  --results-dir /path/to/results/project_x \
  --boltz-prob-threshold 0.65 \
  --boltz-logic50-threshold -0.5 \
  --drug-like Yes \
  --similarity-threshold 0.7
```

Key parameters for `top-hits cluster`:

| Parameter | Meaning |
| --- | --- |
| `--boltz-prob-threshold` | Minimum Boltz-2 binding probability required to keep a top hit for clustering. Default: `0.65`. |
| `--boltz-logic50-threshold` | Maximum Boltz-2 predicted log10(IC50) allowed to keep a top hit for clustering. Default: `-0.5`. |
| `--drug-like` | Required `Drug_like` label. Default: `Yes`. |
| `--similarity-threshold` | Tanimoto similarity cutoff used for Butina clustering. Default: `0.7`. |

## Result Layout

The main result directory is:

```text
<results-dir>/
├── iteration_1/
├── iteration_2/
├── ...
├── model_selection/
└── best_model/
```

Each iteration directory contains the split files, extracted SMILES, extracted fingerprints, affinity prediction outputs, and model outputs for that iteration.

## Top-Hits Cluster Outputs

The `top-hits cluster` step writes its outputs under:

```text
<selection-dir>/top_hits/
```

### Output files

| File | Purpose |
| --- | --- |
| `boltz_rescoring/top_hits_boltz_rescored.tsv` | Merged Boltz-2 rescoring results for all selected top hits |
| `top_hits_annotated.csv` | Top-hits table with FastBindRank predictions, Boltz-2 rescoring results, SMILES, and molecular property annotations |
| `top_hits_clustered.csv` | Annotated table plus threshold-filter result and clustering assignments |
| `top_hits_clustered.summary.log` | Short text summary of clustering statistics |
| `top_hits_clustered.pca.pdf` | PCA visualization of the filtered molecules used for clustering |

### Main columns

| File | Column | Meaning |
| --- | --- | --- |
| `top_hits_boltz_rescored.tsv` | `CID` | Compound identifier carried through the pipeline |
| `top_hits_boltz_rescored.tsv` | `confidence_score` | Overall Boltz-2 confidence score |
| `top_hits_boltz_rescored.tsv` | `ptm` | Predicted TM-score from Boltz-2 confidence output |
| `top_hits_boltz_rescored.tsv` | `iptm` | Interface predicted TM-score from Boltz-2 confidence output |
| `top_hits_boltz_rescored.tsv` | `ligand_iptm` | Ligand-focused interface confidence from Boltz-2 |
| `top_hits_boltz_rescored.tsv` | `affinity_pred_log10_IC50` | Boltz-2 predicted affinity on the log10(IC50) scale |
| `top_hits_boltz_rescored.tsv` | `affinity_probability_binary` | Boltz-2 predicted binding probability |
| `top_hits_annotated.csv` | `rank` | Rank from the full-library FastBindRank prediction list |
| `top_hits_annotated.csv` | `CID` | Compound identifier |
| `top_hits_annotated.csv` | `FastBindRank_pred_prob` | Predicted probability from the selected best FastBindRank model |
| `top_hits_annotated.csv` | `SMILES` | SMILES string for the compound |
| `top_hits_annotated.csv` | `Boltz2_confidence_score` | Renamed Boltz-2 confidence score |
| `top_hits_annotated.csv` | `Boltz2_ptm` | Renamed Boltz-2 pTM value |
| `top_hits_annotated.csv` | `Boltz2_iptm` | Renamed Boltz-2 ipTM value |
| `top_hits_annotated.csv` | `Boltz2_ligand_iptm` | Renamed Boltz-2 ligand ipTM value |
| `top_hits_annotated.csv` | `Boltz2_pred_log10IC50` | Renamed Boltz-2 predicted log10(IC50) |
| `top_hits_annotated.csv` | `Boltz2_pred_prob` | Renamed Boltz-2 predicted binding probability |
| `top_hits_annotated.csv` | `MW` | Molecular weight |
| `top_hits_annotated.csv` | `cLogP` | Calculated logP |
| `top_hits_annotated.csv` | `HBD` | Hydrogen bond donor count |
| `top_hits_annotated.csv` | `HBA` | Hydrogen bond acceptor count |
| `top_hits_annotated.csv` | `TPSA` | Topological polar surface area |
| `top_hits_annotated.csv` | `RotB` | Rotatable bond count |
| `top_hits_annotated.csv` | `RingCount` | Total ring count |
| `top_hits_annotated.csv` | `AromaticRings` | Aromatic ring count |
| `top_hits_annotated.csv` | `HeavyAtoms` | Heavy atom count |
| `top_hits_annotated.csv` | `FormalCharge` | Formal charge |
| `top_hits_annotated.csv` | `Drug_like` | `Yes/No` label based on the current drug-like filter thresholds |
| `top_hits_annotated.csv` | `valid_smiles` | Whether RDKit successfully parsed the SMILES |
| `top_hits_clustered.csv` | `passed_filter` | Whether the molecule passed the user-specified clustering filter thresholds |
| `top_hits_clustered.csv` | `cluster_id` | Butina cluster assignment for filtered molecules |
| `top_hits_clustered.csv` | `cluster_size` | Size of the assigned cluster |
| `top_hits_clustered.csv` | `tanimoto_cutoff` | Similarity cutoff used for clustering |

### Drug-like filter used in annotation

`Drug_like = Yes` currently means:

| Property | Threshold |
| --- | --- |
| `MW` | `250-550` |
| `cLogP` | `1-5` |
| `HBD` | `<= 5` |
| `HBA` | `<= 10` |
| `TPSA` | `<= 140` |
| `RotB` | `<= 10` |
| `RingCount` | `<= 6` |
| `AromaticRings` | `<= 4` |
| `HeavyAtoms` | `20-50` |
| `FormalCharge` | `-1 to +1` |

## Citing This Work

If you use FastBindRank in your research, please cite this work. Citation details will be added here.

## Contributing

We welcome any bug reports, enhancement requests, and other contributions. To submit a bug report or enhancement request, please use the GitHub issues tracker.

## License

This project is licensed under the terms of the MIT license.
