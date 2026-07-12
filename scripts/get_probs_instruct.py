"""Score pairs of sentences with an instruction-tuned Hugging Face causal LM.

Unlike `get_probs.py` (which feeds the bare sentence to the model, base-model
style), this script wraps each sentence in the model's own chat template
behind a fixed instruction, and scores only the sentence tokens conditioned
on that instruction context:

    <chat template>
    user: <instruction>
    assistant: <sentence>              <- only these tokens are scored

This is meant to be run on instruction-tuned models only (see
`utils.models.INSTRUCT_MODELS`), so that they are evaluated the way they are
actually used, rather than as if they were base models. Results are written
to a separate output directory (`../results_instruct/` by default) so they
don't overwrite the base-style runs in `../results/`.
"""

import os
import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple, List

import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm.auto import tqdm

from utils.models import INSTRUCT_MODELS
from utils.data_map import DATA_MAP, DATA_TYPE
from utils.utils import read_ewok, load_model


DATA_PATH = "../data"
BASE_OUTDIR = "../results_instruct/"
TEST_OUTDIR = "../test_instruct/"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

DEFAULT_INSTRUCTION = "Write a natural, coherent English sentence."


@dataclass
class SentenceScore:
	text: str
	logprob: float
	avg_logprob: float


def build_instruction_prefix(tokenizer, instruction: str) -> List[int]:
	"""Render `instruction` through the model's chat template as a user turn,
	with the assistant generation prompt appended, and return its token ids.

	The chat template is responsible for adding whatever special/BOS tokens
	the model expects, so unlike get_probs.py we don't add a BOS manually.
	"""
	messages = [{"role": "user", "content": instruction}]
	encoded = tokenizer.apply_chat_template(
		messages, tokenize=True, add_generation_prompt=True, return_dict=True
	)
	# `return_dict=True` is requested explicitly and pulled by key, rather than
	# relying on the default return type: depending on the transformers version,
	# tokenize=True alone can hand back either a plain list of ints or a
	# BatchEncoding, and `list(a_batch_encoding)` silently yields its string
	# keys instead of token ids.
	ids = encoded["input_ids"]
	if len(ids) > 0 and isinstance(ids[0], (list, tuple)):
		ids = ids[0]  # unwrap a batch dimension if the template returned one
	return [int(i) for i in ids]


def sentence_logprobs_by_layer(
	model, tokenizer, sentence: str, prefix_ids: List[int], device: str = None
) -> list[SentenceScore]:
	sentence_ids = tokenizer.encode(sentence, add_special_tokens=False)
	input_ids = prefix_ids + sentence_ids
	prefix_len = len(prefix_ids)

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

		# Only score the sentence tokens, conditioned on the instruction prefix:
		# shift index (prefix_len - 1) predicts the first sentence token.
		sent_shift_logits = shift_logits[:, prefix_len - 1:, :]
		sent_shift_labels = shift_labels[:, prefix_len - 1:]

		token_logprobs = torch.log_softmax(sent_shift_logits, dim=-1).gather(
			2, sent_shift_labels.unsqueeze(-1)
		).squeeze(-1)

		total_logprob = token_logprobs.sum().item()
		avg_logprob = token_logprobs.mean().item() if token_logprobs.numel() else float("nan")
		layer_scores.append(SentenceScore(sentence, total_logprob, avg_logprob))

	return layer_scores


def score_pair(
	model, tokenizer, first: str, second: str, prefix_ids: List[int], device: str = None
) -> Tuple[list[SentenceScore], list[SentenceScore]]:
	first_scores = sentence_logprobs_by_layer(model, tokenizer, first, prefix_ids, device)
	second_scores = sentence_logprobs_by_layer(model, tokenizer, second, prefix_ids, device)
	return first_scores, second_scores


def score_dataset(
	model, tokenizer, data: pd.DataFrame, out_path: Path, label: str,
	prefix_ids: List[int], instruction: str,
) -> None:
	rows_by_layer = None
	for row in tqdm(data.itertuples(), total=len(data)):
		first_scores, second_scores = score_pair(
			model, tokenizer, row.sentence_good, row.sentence_bad, prefix_ids, device=DEVICE
		)
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
					"instruction": instruction,
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
	return pd.read_json(file_path, lines=True)[['sentence_good', 'sentence_bad', 'field', 'linguistics_term']]


def main() -> None:
	parser = argparse.ArgumentParser(
		description="Score sentence pairs with an instruction-tuned causal LM, conditioned on a chat-template instruction."
	)
	parser.add_argument("--data", required=True, help="Dataset key to score")
	parser.add_argument("--model", required=True, choices=list(INSTRUCT_MODELS.keys()), help="Instruction-tuned model name")
	parser.add_argument("--test", required=False, action="store_true", help="Use only the first 10 examples")
	parser.add_argument("--ewok-root", required=False, help="Optional root path to EWoK output dataset files")
	parser.add_argument(
		"--instruction",
		default=DEFAULT_INSTRUCTION,
		help="Instruction placed in the chat template's user turn before each sentence is scored as the assistant turn.",
	)
	args = parser.parse_args()

	dtype = DATA_TYPE.get(args.data, "blimp")
	model_id = INSTRUCT_MODELS[args.model]

	print(f"Loading {model_id}...")
	model, tokenizer = load_model(model_id, device=DEVICE)

	prefix_ids = build_instruction_prefix(tokenizer, args.instruction)
	print(f"Instruction prefix ({len(prefix_ids)} tokens): {tokenizer.decode(prefix_ids)!r}")

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
			score_dataset(model, tokenizer, data, out_path, f"ewok-{suite_name}", prefix_ids, args.instruction)

		return

	data_path = Path(DATA_PATH) / DATA_MAP[args.data]
	data = read_blimp(data_path)
	if args.test:
		data = data.head(10)

	out_path = Path(TEST_OUTDIR if args.test else BASE_OUTDIR) / args.model / args.data
	os.makedirs(out_path, exist_ok=True)
	score_dataset(model, tokenizer, data, out_path, args.data, prefix_ids, args.instruction)


if __name__ == "__main__":
	main()
