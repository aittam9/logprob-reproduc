"""Score pairs of sentences with a causal language model, layer by layer.

The script reads a TSV file with two text columns, computes the logprob of
each sentence at every layer of the model under a logit lens, and reports
whether the first sentence is more likely than the second.

Model internals are accessed through NNsight (https://nnsight.net): every
forward pass runs inside an `nnsight.LanguageModel` trace, and layer
activations are read off via `LanguageModel.output.hidden_states` rather
than manual hooks. Pass `--remote` to run those traces on NDIF instead of
locally (see `--help` and the README for the required API key).
"""

import os
import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import pandas as pd
from tqdm.auto import tqdm

from utils.models import BASE_MODELS, INSTRUCT_MODELS
from utils.data_map import DATA_MAP, DATA_TYPE
from utils.utils import read_blimp, read_ewok, load_model, layer_logprobs, configure_ndif


ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT_DIR / "data"
BASE_OUTDIR = ROOT_DIR / "results"
TEST_OUTDIR = ROOT_DIR / "test"


@dataclass
class SentenceScore:
	text: str
	logprob: float
	avg_logprob: float


def sentence_logprobs_by_layer(model, tokenizer, sentence: str, remote: bool = False) -> list[SentenceScore]:
	prefix_ids = [tokenizer.bos_token_id] if tokenizer.bos_token_id is not None else []
	continuation_ids = tokenizer.encode(sentence, add_special_tokens=False)

	scores = layer_logprobs(model, prefix_ids, continuation_ids, remote=remote)
	return [SentenceScore(sentence, total, avg) for total, avg in scores]


def score_pair(model, tokenizer, first: str, second: str, remote: bool = False) -> Tuple[list[SentenceScore], list[SentenceScore]]:
	first_scores = sentence_logprobs_by_layer(model, tokenizer, first, remote=remote)
	second_scores = sentence_logprobs_by_layer(model, tokenizer, second, remote=remote)
	return first_scores, second_scores


def score_dataset(model, tokenizer, data: pd.DataFrame, out_path: Path, label: str, remote: bool = False) -> None:
	rows_by_layer = None
	for row in tqdm(data.itertuples(), total=len(data)):
		first_scores, second_scores = score_pair(model, tokenizer, row.sentence_good, row.sentence_bad, remote=remote)
		if rows_by_layer is None:
			rows_by_layer = [[] for _ in range(len(first_scores))]
		for layer_index, (first_score, second_score) in enumerate(zip(first_scores, second_scores), start=1):
			rows_by_layer[layer_index - 1].append(
				{
					"sent_id": row.Index,
					"layer": layer_index,
					"sentence_good": first_score.text,
					"sentence_bad": second_score.text,
					"sent1_logprob": first_score.logprob,
					"sent2_logprob": second_score.logprob,
					"sent1_avg_logprob": first_score.avg_logprob,
					"sent2_avg_logprob": second_score.avg_logprob,
					"sent1_logprob_greater": first_score.logprob > second_score.logprob,
					"sent1_avg_logprob_greater": first_score.avg_logprob > second_score.avg_logprob,
					"field": row.field,
					"linguistics_term": row.linguistics_term,
				}
			)

	print()
	result = pd.DataFrame([row for rows in rows_by_layer for row in rows])
	print(
		f"First sentence has higher logprob in {result['sent1_logprob_greater'].mean() * 100:.2f}% of pairs across all layers for {label}"
	)
	outfile_name = f"{label}_scored"
	result.to_csv(f"{out_path}/{outfile_name}.tsv", index=False, sep="\t")
	print(f"Results saved to {out_path}/{outfile_name}.tsv")

	print("\n\n-----------------------------------Summary of results by layer:--------------------------------------")
	for layer_index, layer_df in result.groupby("layer"):
		layer_accuracy = layer_df["sent1_logprob_greater"].mean() * 100
		print(f"Layer {layer_index}: First sentence has higher logprob in {layer_accuracy:.2f}% of pairs")


def main() -> None:
	parser = argparse.ArgumentParser(description="Score sentence pairs with a causal LM, layer by layer, via NNsight.")
	parser.add_argument("--data", required=True, help="Dataset key to score")
	parser.add_argument("--model", required=True, choices=list(BASE_MODELS.keys()) + list(INSTRUCT_MODELS.keys()), help="Model name")
	parser.add_argument("--test", required=False, action="store_true", help="Use only the first 10 examples")
	parser.add_argument("--ewok-root", required=False, help="Optional root path to EWoK output dataset files")
	parser.add_argument("--device", required=False, default=None, help="Device to load the model on (default: cuda if available, else cpu). Ignored with --remote.")
	parser.add_argument("--remote", required=False, action="store_true", help="Run model traces on NDIF instead of locally; no local weights are downloaded.")
	parser.add_argument("--ndif-api-key", required=False, default=None, help="NDIF API key for --remote; defaults to the NDIF_API_KEY environment variable.")
	args = parser.parse_args()

	if args.remote:
		configure_ndif(args.ndif_api_key)

	dtype = DATA_TYPE.get(args.data, "blimp")
	if args.model in BASE_MODELS:
		model_id = BASE_MODELS[args.model]
	elif args.model in INSTRUCT_MODELS:
		model_id = INSTRUCT_MODELS[args.model]
	else:
		raise ValueError(f"Model {args.model} not found in either BASE_MODELS or INSTRUCT_MODELS")

	print(f"Loading {model_id}{' (remote via NDIF)' if args.remote else ''}...")
	model, tokenizer = load_model(model_id, device=args.device, remote=args.remote)

	if dtype == "ewok":
		if args.ewok_root:
			ewok_root = Path(args.ewok_root)
		else:
			ewok_root = ROOT_DIR / "ewok-paper" / "output" / "dataset"

		if not ewok_root.exists():
			raise FileNotFoundError(f"EWoK output directory not found at {ewok_root}; place built EWoK outputs there or adjust path.")

		testsuite_files = sorted(ewok_root.rglob("testsuite-*.csv"))
		if not testsuite_files:
			raise FileNotFoundError(f"No testsuite-*.csv files found under {ewok_root}")

		for testsuite_path in testsuite_files:
			print(f"Processing EWoK testsuite: {testsuite_path}")
			data = read_ewok(testsuite_path)
			if args.test:
				data = data.head(10)

			suite_name = testsuite_path.stem.replace("testsuite-", "")
			out_path = (TEST_OUTDIR if args.test else BASE_OUTDIR) / args.model / f"ewok-{suite_name}"
			os.makedirs(out_path, exist_ok=True)
			score_dataset(model, tokenizer, data, out_path, f"ewok-{suite_name}", remote=args.remote)

		return

	data_path = DATA_PATH / DATA_MAP[args.data]
	data = read_blimp(data_path)
	if args.test:
		data = data.head(10)

	out_path = (TEST_OUTDIR if args.test else BASE_OUTDIR) / args.model / args.data
	os.makedirs(out_path, exist_ok=True)
	score_dataset(model, tokenizer, data, out_path, args.data, remote=args.remote)


if __name__ == "__main__":
	main()
