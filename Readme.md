# BLIMP Probability Runs

Use `scripts/run_get_probs.sh` to batch `scripts/get_probs.py` across models and datasets.

```bash
bash scripts/run_get_probs.sh
```

Runs every model on every dataset.

```bash
bash scripts/run_get_probs.sh test
```

Runs the full sweep in test mode, using only the first 10 examples per dataset.

```bash
bash scripts/run_get_probs.sh all_models data 3
```

Runs all models on dataset `3` only.

```bash
bash scripts/run_get_probs.sh model Llama-3.2-1B all_data
```

Runs one model across all datasets.

```bash
bash scripts/run_get_probs.sh test model Llama-3.2-1B data 3
```

Runs one model on one dataset in test mode.

Model names come from `scripts/utils/models.py`, and dataset ids come from `scripts/utils/data_map.py`.