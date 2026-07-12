"""Score pairs of sentences with a Hugging Face causal language model.

The script reads a TSV file with two text columns, computes the logprob of each
sentence under a causal LM, and reports whether the first sentence is more
likely than the second.
"""

import os
import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm.auto import tqdm 

from utils.models import BASE_MODELS, INSTRUCT_MODELS
from utils.data_map import DATA_MAP, DATA_TYPE
from utils.utils import read_ewok, load_model


# CACHE_DIR = "/extra/mattia.proietti/hf_models"
DATA_PATH = "../data"
BASE_OUTDIR = "../results/"
TEST_OUTDIR = "../test/"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"



@dataclass
class SentenceScore:
	text: str
	logprob: float
	avg_logprob: float


def sentence_logprobs_by_layer(model, tokenizer, sentence: str, device: str = None) -> list[SentenceScore]:
	input_ids = tokenizer.encode(sentence, add_special_tokens=False)
	if tokenizer.bos_token_id is not None:
		input_ids = [tokenizer.bos_token_id] + input_ids

	inputs = {
		"input_ids": torch.tensor([input_ids], device=device),
		"attention_mask": torch.ones((1, len(input_ids)), device=device, dtype=torch.long),
	}

	output_head = model.get_output_embeddings()
	if output_head is None and hasattr(model, "lm_head"):
		output_head = model.lm_head

	with torch.no_grad():
		outputs = model(**inputs, output_hidden_states=True, return_dict=True)

	layer_scores = []
	for hidden_state in outputs.hidden_states[1:]:
		logits = output_head(hidden_state)
		shift_logits = logits[:, :-1, :]
		shift_labels = inputs["input_ids"][:, 1:]

		token_logprobs = torch.log_softmax(shift_logits, dim=-1).gather(
			2, shift_labels.unsqueeze(-1)
		).squeeze(-1)

		total_logprob = token_logprobs.sum().item()
		avg_logprob = token_logprobs.mean().item() if token_logprobs.numel() else float("nan")
		layer_scores.append(SentenceScore(sentence, total_logprob, avg_logprob))

	return layer_scores


def score_pair(model, tokenizer, first: str, second: str, device: str = None) -> Tuple[list[SentenceScore], list[SentenceScore]]:
	first_scores = sentence_logprobs_by_layer(model, tokenizer, first, device)
	second_scores = sentence_logprobs_by_layer(model, tokenizer, second, device)
	return first_scores, second_scores


def score_dataset(model, tokenizer, data: pd.DataFrame, out_path: Path, label: str) -> None:
	rows_by_layer = None
	for row in tqdm(data.itertuples(), total=len(data)):
		first_scores, second_scores = score_pair(model, tokenizer, row.sentence_good, row.sentence_bad, device=DEVICE)
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


def read_blimp(file_path):  
    return pd.read_json(file_path, lines=True)[['sentence_good', 'sentence_bad','field', 'linguistics_term']]


def main() -> None:
	parser = argparse.ArgumentParser(description="Score sentence pairs with a causal LM.")
	parser.add_argument("--data", required=True, help="Dataset key to score")
	parser.add_argument("--model", required=True, choices=list(BASE_MODELS.keys()) + list(INSTRUCT_MODELS.keys()), help="Hugging Face causal LM name or path")
	parser.add_argument("--test", required=False, action="store_true", help="Use only the first 10 examples")
	parser.add_argument("--ewok-root", required=False, help="Optional root path to EWoK output dataset files")
	args = parser.parse_args()

	dtype = DATA_TYPE.get(args.data, "blimp")
	if args.model in BASE_MODELS:
		model_id = BASE_MODELS[args.model]
	elif args.model in INSTRUCT_MODELS:
		model_id = INSTRUCT_MODELS[args.model]
	else:
		raise ValueError(f"Model {args.model} not found in either BASE_MODELS or INSTRUCT_MODELS")

	print(f"Loading {model_id}...")
	model, tokenizer = load_model(model_id, device=DEVICE)

	if dtype == "ewok":
		if args.ewok_root:
			ewok_root = Path(args.ewok_root)
		else:
			ewok_root = Path(__file__).resolve().parents[1] / "ewok-paper" / "output" / "dataset"
			if not ewok_root.exists():
				ewok_root = Path("../ewok-paper/output/dataset").resolve()

		if not ewok_root.exists():
			raise FileNotFoundError(f"EWoK output directory not found at {ewok_root}; place built EWoK outputs there or adjust path.")

		testsuite_files = sorted(ewok_root.rglob("testsuite-*.csv"))
		if not testsuite_files:
			testsuite_files = sorted(ewok_root.glob("**/testsuite-*.csv"))
		if not testsuite_files:
			raise FileNotFoundError(f"No testsuite-*.csv files found under {ewok_root}")

		for testsuite_path in testsuite_files:
			print(f"Processing EWoK testsuite: {testsuite_path}")
			data = read_ewok(testsuite_path)
			if args.test:
				data = data.head(10)

			suite_name = testsuite_path.stem.replace("testsuite-", "")
			out_path = Path(TEST_OUTDIR if args.test else BASE_OUTDIR) / args.model / f"ewok-{suite_name}"
			os.makedirs(out_path, exist_ok=True)
			score_dataset(model, tokenizer, data, out_path, f"ewok-{suite_name}")

		return

	data_path = Path(DATA_PATH) / DATA_MAP[args.data]
	data = read_blimp(data_path)
	if args.test:
		data = data.head(10)

	out_path = Path(TEST_OUTDIR if args.test else BASE_OUTDIR) / args.model / args.data
	os.makedirs(out_path, exist_ok=True)
	score_dataset(model, tokenizer, data, out_path, args.data)


if __name__ == "__main__":
	main()
