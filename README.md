<h1 align="center">GUTS</h1>

<p align="center">
  <b>Efficient Large-Scale Graph Unlearning through Trajectory-Aware Sharding</b>
</p>

<p align="center">
  <a href="#overview">Overview</a> ·
  <a href="#requirements">Requirements</a> ·
  <a href="#experiment-notebooks">Notebooks</a> ·
  <a href="#yjmob">YJMob</a> ·
  <a href="#cora">Cora</a> ·
  <a href="#configuration">Configuration</a>
</p>

## Overview

GUTS is a shard-based graph unlearning framework for trajectory data: a deletion request
retrains only the affected shards, and GUTS partitions the graph so that few shards are affected.
The code covers **YJMob** (next-location prediction) and **Cora** (node classification).

| Component | Paper | Code |
|---|:---:|---|
| Deletion Influence | §4.1 | `guts/method/partition/partitioner.py` |
| Hypergraph partitioning (KaHyPar) | §4.2 | `guts/method/partition/partitioner.py` |
| Boundary Refinement | §4.3 | `guts/method/partition/refinement.py` |
| Temporal Subgraph Assignment | §4.4 | `guts/method/partition/assignment.py` |

![GUTS pipeline overview](assets/pipeline_overview.png)

The figures in the paper can be reproduced with the notebooks in `Analysis/`.

## Repository layout

```
configs/                 YAML configs (yjmob_14d.yaml, cora.yaml)
experiments_*.ipynb      experiment notebooks
guts/method/             GUTS partitioner and shared code
guts/yjmob/              YJMob pipeline, Full-Graph baseline, unlearning
guts/static_graph/       Cora pipeline, Full-Graph baseline, unlearning
Analysis/                notebooks for the paper figures
```

## Requirements

```
python 3.9
torch 1.13.1 (CUDA 11.7)   torch_geometric 2.0.4   kahypar 1.3.6
numpy 1.26   scipy 1.13   pandas 2.3   scikit-learn 1.6   pyyaml 6.0
jupyter (for the experiment notebooks)
```

## Experimental environment

> - **OS** Ubuntu 22.04.5 LTS
> - **CPU** Intel Core i9-10980XE (18 cores, 36 threads)
> - **GPU** 4 × NVIDIA GeForce RTX 3080 (10 GB)
> - **Memory** 128 GB

## Experiment notebooks

The notebooks run a whole experiment (data, Full-Graph baseline, GUTS, shard unlearning, result
tables) with the commands below. Start Jupyter from the repository root.

- `experiments_yjmob.ipynb` — YJMob
- `experiments_cora.ipynb` — Cora

## YJMob

[YJMob100K](https://zenodo.org/records/10142719) holds 75 days of anonymized mobility of 100,000
users on a 200 × 200 grid. Put `yjmob100k-dataset1.csv` (decompressed, sorted by `uid`) in
`processed/yjmob/raw/`. Run every command from the repository root.

**14-day dataset** (`processed/yjmob_1_14d`)

```bash
python guts/yjmob/data/preprocess.py --dataset_name dataset1 --day_range 0 13 --out_name yjmob_1_14d
```

**GUTS run** — partition, shard training, evaluation (`Results/shard-runs/yjmob_14d/`)

```bash
python guts/yjmob/pipeline.py --config configs/yjmob_14d.yaml --num-gpus 4
```

**Full-Graph baseline** (`Results/runs/yjmob_14d/`)

```bash
python guts/yjmob/full_graph.py --config configs/yjmob_14d.yaml --num-gpus 4
```

**Forget sets** (`processed/forget_sets/yjmob/`)

```bash
python guts/yjmob/unlearning/forget_sets.py strata --source_dir processed/yjmob_1_14d
python guts/yjmob/unlearning/forget_sets.py batch --source_dir processed/yjmob_1_14d \
    --types user node edge traj --percent 0.05 --seed 2026
```

**Clean data** without a forget set (`--exclude_{users,nodes,edges,trajs}_from`)

```bash
python guts/yjmob/data/preprocess.py --dataset_name dataset1 --day_range 0 13 \
    --fixed_grid_info_from processed/yjmob_1_14d/next_location_meta.pt \
    --exclude_nodes_from <forget_set>.json --out_name yjmob_1_14d_clean_<tag>
```

**Retraining baselines** on the clean data — Shard-Retrain with `pipeline.py`, Full-Retrain with `full_graph.py`

```bash
python guts/yjmob/pipeline.py --config configs/yjmob_14d.yaml --num-gpus 4 \
    --graphsage-source-processed-dir processed/yjmob_1_14d_clean_<tag> --seed <seed>
```

**Shard unlearning** of a GUTS run (`Results/shard-unlearns/yjmob_14d/`)

```bash
python guts/yjmob/unlearning/unlearn_batch.py --state_dir Results/shard-runs/yjmob_14d/<run> \
    --forget_type node --forget_set_path <forget_set>.json \
    --clean_source_dir processed/yjmob_1_14d_clean_<tag> --num_gpus 4
```

## Cora

Each run uses one seed and one model (`graphsage`, `gcn`, `gat` or `gin`); `experiments_cora.ipynb`
repeats them over seeds and models. The Cora dataset is downloaded by PyG on first use.

**Split file** — stratified 80/20 train/test split (`processed/Cora_split/cora_seed2026.pt`)

```bash
python guts/static_graph/cora_data.py split --seed 2026
```

**Forget sets** drawn from the train part (`processed/forget_sets/Cora/{Node,Edge}/`)

```bash
python guts/static_graph/forget_sets.py --data_path processed/Cora_split/cora_seed2026.pt \
    --types node edge --percent 0.5 --seed 2026
```

**GUTS run** — partition the train graph, one model per shard, evaluation on the full graph

```bash
python guts/static_graph/pipeline.py --config configs/cora.yaml \
    --data_path processed/Cora_split/cora_seed2026.pt --model_type graphsage --seed 2026
```

**Full-Graph baseline** — one model on the train graph

```bash
python guts/static_graph/full_graph.py --config configs/cora.yaml \
    --data_path processed/Cora_split/cora_seed2026.pt --model_type graphsage --seed 2026
```

**Shard unlearning** of a GUTS run

```bash
python guts/static_graph/unlearn.py --run_dir Results/shard-runs/Cora/<run> \
    --forget_set processed/forget_sets/Cora/Node/forget_node_seed2026_0.5pct.json
```

The test prediction averages the softmax outputs of all shard models (accuracy and macro F1).
Unlearning retrains the affected shards without the deleted nodes or edges, keeps the other
shard models, and evaluates on the clean graph built in memory. For the retraining baselines,
build the clean graph with `cora_data.py clean --data_path <split file> --forget_set <forget_set>.json`
and pass it as `--data_path` to `pipeline.py` (Shard-Retrain) or `full_graph.py` (Full-Retrain).
Results are written to `Results/{shard-runs,runs,shard-unlearns}/Cora/<run>/`.

## Configuration

Main keys of `configs/*.yaml`, with their symbols in the paper:

| Key | Meaning |
|---|---|
| `partition.num_shards` | number of shards K |
| `partition.epsilon` | KaHyPar node-imbalance tolerance ε |
| `partition.di_alpha`, `di_beta` | weights α, β of TF and UC\* in the Deletion Influence |
| `partition.refine_epsilon` | shard-size limit during Boundary Refinement: (1 + `refine_epsilon`) × the average shard size of the day (1.0 = twice the average) |
| `partition.lambda_di`, `lambda_c`, `lambda_l` | refinement weights λ<sub>DI</sub>, λ<sub>C</sub>, λ<sub>ℓ</sub> (0 turns a term off) |
| `partition.use_temporal_subgraph_assignment` | Temporal Subgraph Assignment |
| `temporal_graph.window_size` | days merged into a target day's graph |
| `training.seed` | run seed (random when absent) |
| `training.*` | per-shard model and training hyperparameters (YJMob) |

Any key can be overridden on the command line:

```bash
--override partition.num_shards=64 training.epochs=50
```

## Note

Parts of this code were written with the assistance of generative AI tools and reviewed by the authors.
