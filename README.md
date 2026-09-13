# IBOT-NA

IBOT-NA is an information-constrained optimal transport method for network alignment. It combines a shared residual encoder, a variational information bottleneck, and fused Gromov-Wasserstein optimal transport.

## Repository layout

```text
IBOT-NA-code2/
|-- config/                    # Dataset-specific hyperparameters
|-- data/                      # Paired-network datasets in NPZ format
|-- models/IBOT-NA/            # Model and training program
|-- scripts/                   # Main-experiment entry points
|-- src/ibot_na/               # Data, execution, and reporting modules
|-- algorithm.pdf              # Algorithm description
|-- LICENSE
|-- pyproject.toml
`-- requirements.txt
```

Experiment records, logs, and runtime caches are written to `Temp/`. The summary workbook is written to `result/main_results.xlsx`.

## Environment

The reference environment uses Python 3.12.3 and the package versions in `requirements.txt`. From the repository root, run:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -e . --no-deps
```

The requirements file selects the PyTorch CUDA 13.0 wheel index. For CPU execution, install the corresponding CPU build of PyTorch and pass `--device cpu` to the experiment script.

## Data

The `data/` directory contains seven paired-network datasets:

```text
foursquare-twitter
phone-email
arxiv
GGI
douban
ACM-DBLP
dbp15k_zh-en
```

Files use the name `<dataset>-pa_0.2.npz`. Dataset lookup is case-insensitive, including the file `Douban-pa_0.2.npz`.

Each file contains:

| Key | Description | Shape |
|---|---|---|
| `edge_index1` | Edges in network 1 | `(2, E1)` or `(E1, 2)` |
| `edge_index2` | Edges in network 2 | `(2, E2)` or `(E2, 2)` |
| `edge_attr1` | Edge attributes in network 1 | `(E1, De1)` |
| `edge_attr2` | Edge attributes in network 2 | `(E2, De2)` |
| `pos_pairs` | Supervised alignment pairs | `(Ns, 2)` |
| `test_pairs` | Test alignment pairs | `(Nt, 2)` |
| `num_nodes1` | Number of nodes in network 1 | scalar |
| `num_nodes2` | Number of nodes in network 2 | scalar |

Attributed datasets also contain `x1` and `x2`, with one feature row per node.

## Main experiment

The stored alignment pairs contain a 20% supervised pool and an 80% test set. For each seed, a seeded CPU `torch.randperm` divides the supervised pool equally between training and validation. This gives a 10% training, 10% validation, and 80% test protocol. Validation MRR selects the checkpoint used for test evaluation.

The default seeds are `0 1 2 3 4`. Run the complete seven-dataset experiment with:

```bash
python scripts/run_all.py --device cuda
```

Run one or more selected datasets with:

```bash
python scripts/run_experiment.py --datasets foursquare-twitter GGI --device cuda
```

Use `--dry-run` to inspect the resolved datasets, seeds, parameters, and paths before training:

```bash
python scripts/run_all.py --device cuda --dry-run
```

CPU execution uses the same interface:

```bash
python scripts/run_experiment.py --datasets phone-email --device cpu
```

## Configuration

Dataset-specific values are stored in `config/ibot_na_hyperparams.json`. The configurable fields are:

```text
alpha
gamma_p
init_threshold_lambda
ib_kl_weight
ib_anchor_weight
lr
epochs
```

Command-line values take precedence over the JSON configuration. For example:

```bash
python scripts/run_experiment.py --datasets ACM-DBLP --alpha 0.8 --override gamma_p=0.002 --device cuda
```

Repeated `--override KEY=VALUE` arguments can set other training-program options.

## Results and resuming

Seed-level records are appended to `Temp/experiment_records.jsonl` and mirrored in `Temp/experiment_records.json`. Each run also writes its manifest and process logs below `Temp/`.

`result/main_results.xlsx` contains one worksheet per dataset. Hits@1, Hits@10, MRR, runtime, and memory are reported as the mean and population standard deviation across the five seeds.

Completed dataset-seed records are reused when a command is resumed. Pass `--rerun` to execute them again. The default time limit is 3600 seconds per seed and can be reduced with `--timeout-s`.

## License

This project is distributed under the MIT License. See `LICENSE`.
