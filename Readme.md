# Layer-wise logprob probing on BLiMP / EWoK

Scores BLiMP- and EWoK-style sentence pairs with causal language models and
reports, at every layer, which sentence of the pair the model finds more
likely under a logit lens (the model's own output head applied to each
layer's hidden state). Two scoring modes live in `scripts/`:

- `get_probs.py` -- compares the two sentences' raw logprobs, base-model style.
- `get_probs_instruct.py` -- same comparison, but each sentence is scored as
  an assistant turn following a fixed instruction in the model's chat
  template (instruction-tuned models only).

## Model internals via NNsight

All access to model internals goes through
[NNsight](https://nnsight.net): every forward pass runs inside an
`nnsight.LanguageModel` trace (`scripts/utils/utils.py::layer_logprobs`), and
per-layer hidden states are read off `LanguageModel.output.hidden_states`
rather than manual forward hooks. This is what makes the same code path work
locally and remotely.

Pass `--remote` to either script (or `remote` to their
`run_get_probs*.sh` wrappers) to run those traces on
[NDIF](https://ndif.us) instead of on local hardware. In remote mode no model
weights are downloaded locally at all -- only the tokenizer and config are
needed on your machine, and the trace itself (and the small per-layer
logprob scalars it returns) run on NDIF's infrastructure. This requires an
NDIF API key:

1. Request one at https://login.ndif.us.
2. Set it as an environment variable: `export NDIF_API_KEY=...` (or pass
   `--ndif-api-key` explicitly).

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Some of the models in `scripts/utils/models.py` (e.g. Llama, Gemma) are
gated on Hugging Face Hub. Authenticate first with `huggingface-cli login`
or by setting `HF_TOKEN`; this isn't needed at all if you're running
everything through `--remote`/NDIF.

### Data

- **BLiMP**: place the per-phenomenon `*.jsonl` files (see
  `scripts/utils/data_map.py` for the expected filenames) under `data/`.
- **EWoK**: build the [ewok-paper](https://github.com/ewok-core/ewok-paper)
  dataset and point `--ewok-root` at its `output/dataset` directory (by
  default the scripts look for `ewok-paper/output/dataset` next to this
  repo).

Neither dataset is included in this repository.

## Usage

Use `scripts/run_get_probs.sh` to batch `scripts/get_probs.py` across models
and datasets (`get_probs_instruct.py` has a matching
`run_get_probs_instruct.sh` wrapper with the same argument style).

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

```bash
bash scripts/run_get_probs.sh model Llama-3.2-1B all_data remote
```

Runs one model across all datasets with the traces executed remotely on NDIF.

Model names come from `scripts/utils/models.py`, and dataset ids come from
`scripts/utils/data_map.py`. Each script can also be run directly, e.g.
`python scripts/get_probs.py --model Llama-3.2-1B --data 3 --remote`; see
`--help` on any of them for the full set of flags.

Results are written to `results/` and `results_instruct/` respectively
(`test/` and `test_instruct/` in test mode).
