"""Score BLiMP/EWoK pairs via a forced 1-or-2 multiple-choice prompt.

Unlike `get_probs.py` / `get_probs_instruct.py` (which compare the two
sentences' own logprobs directly, with no task framing), this script puts
both sentences of a pair into a single prompt and asks the model to answer
"1" or "2":

    Here are two English sentences: 1) <sentence 1> 2) <sentence 2> Which
    sentence is more plausible? Respond with either 1 or 2 as your answer.
    Answer: <1 or 2>

and reads off which answer token the model assigns higher probability to,
at every layer (same logit-lens approach as the other two scripts).
`sentence_good` is always placed in position 1, so `choice1_logprob_greater`
gives per-layer accuracy directly.

Works for both base and instruction-tuned models (see `utils.models.
BASE_MODELS` / `INSTRUCT_MODELS`): instruction-tuned models get the prompt
wrapped in their chat template as a user turn (so the "1"/"2" is scored as
an actual assistant reply); base models get it as plain continued text (BOS
+ prompt), matching the convention `get_probs.py` uses. Results are written
to `../results_choice/` by default so they don't collide with either of the
other two result directories.

Note: sentence_good is always option 1, so a model with a position bias
(e.g. always favoring "1") would inflate accuracy here; a natural follow-up
would be to also run with the order swapped and average the two.
"""

import os
import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm.auto import tqdm

from utils.models import BASE_MODELS, INSTRUCT_MODELS
from utils.data_map import DATA_MAP, DATA_TYPE
from utils.utils import read_ewok, load_model


DATA_PATH = "../data"
BASE_OUTDIR = "../results_choice/"
TEST_OUTDIR = "../test_choice/"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

PROMPT_TEMPLATE = (
	"Here are two English sentences: 1) {sent1} 2) {sent2} Which sentence is "
	"more plausible? Respond with either 1 or 2 as your answer. Answer:"
)
CHOICE_CONTINUATIONS = {"1": " 1", "2": " 2"}


@dataclass
class ContinuationScore:
	text: str
	logprob: float
	avg_logprob: float


def build_prefix_ids(tokenizer, prompt_text: str, is_instruct: bool) -> List[int]:
	"""Tokenize the choice prompt as a chat-template user turn (instruction-
	tuned models) or as plain BOS-prefixed text (base models)."""
	if is_instruct:
		messages = [{"role": "user", "content": prompt_text}]
		encoded = tokenizer.apply_chat_template(
			messages, tokenize=True, add_generation_prompt=True, return_dict=True
		)
		# see get_probs_instruct.py::build_instruction_prefix for why we pull
		# input_ids by key and unwrap a batch dim rather than trusting the
		# return type of tokenize=True across transformers versions
		ids = encoded["input_ids"]
		if len(ids) > 0 and isinstance(ids[0], (list, tuple)):
			ids = ids[0]
		return [int(i) for i in ids]

	ids = tokenizer.encode(prompt_text, add_special_tokens=False)
	if tokenizer.bos_token_id is not None:
		ids = [tokenizer.bos_token_id] + ids
	return ids


def continuation_logprobs_by_layer(
	model, tokenizer, continuation: str, prefix_ids: List[int], device: str = None
) -> List[ContinuationScore]:
	continuation_ids = tokenizer.encode(continuation, add_special_tokens=False)
	input_ids = prefix_ids + continuation_ids
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

		# Only score the continuation tokens ("1"/"2"), conditioned on the
		# prompt: shift index (prefix_len - 1) predicts the first
		# continuation token.
		cont_shift_logits = shift_logits[:, prefix_len - 1:, :]
		cont_shift_labels = shift_labels[:, prefix_len - 1:]

		token_logprobs = torch.log_softmax(cont_shift_logits, dim=-1).gather(
			2, cont_shift_labels.unsqueeze(-1)
		).squeeze(-1)

		total_logprob = token_logprobs.sum().item()
		avg_logprob = token_logprobs.mean().item() if token_logprobs.numel() else float("nan")
		layer_scores.append(ContinuationScore(continuation, total_logprob, avg_logprob))

	return layer_scores


def score_choice(
	model, tokenizer, prefix_ids: List[int], device: str = None
) -> Tuple[List[ContinuationScore], List[ContinuationScore]]:
	choice1_scores = continuation_logprobs_by_layer(model, tokenizer, CHOICE_CONTINUATIONS["1"], prefix_ids, device)
	choice2_scores = continuation_logprobs_by_layer(model, tokenizer, CHOICE_CONTINUATIONS["2"], prefix_ids, device)
	return choice1_scores, choice2_scores


def score_dataset(
	model, tokenizer, data: pd.DataFrame, out_path: Path, label: str, is_instruct: bool,
) -> None:
	rows_by_layer = None
	for row in tqdm(data.itertuples(), total=len(data)):
		prompt_text = PROMPT_TEMPLATE.format(sent1=row.sentence_good, sent2=row.sentence_bad)
		prefix_ids = build_prefix_ids(tokenizer, prompt_text, is_instruct)
		choice1_scores, choice2_scores = score_choice(model, tokenizer, prefix_ids, device=DEVICE)
		if rows_by_layer is None:
			rows_by_layer = [[] for _ in range(len(choice1_scores))]
		for layer_index, (c1, c2) in enumerate(zip(choice1_scores, choice2_scores), start=1):
			rows_by_layer[layer_index - 1].append(
				{
					"sent_id": row.Index,
					"layer": layer_index,
					"sentence_good": row.sentence_good,
					"sentence_bad": row.sentence_bad,
					"choice1_logprob": c1.logprob,
					"choice2_logprob": c2.logprob,
					"choice1_avg_logprob": c1.avg_logprob,
					"choice2_avg_logprob": c2.avg_logprob,
					"choice1_logprob_greater": c1.logprob > c2.logprob,
					"choice1_avg_logprob_greater": c1.avg_logprob > c2.avg_logprob,
					"field": row.field,
					"linguistics_term": row.linguistics_term,
				}
			)

	print()
	result = pd.DataFrame([row for rows in rows_by_layer for row in rows])
	print(
		f"Choice '1' (the plausible sentence) has higher logprob in {result['choice1_logprob_greater'].mean() * 100:.2f}% of pairs across all layers for {label}"
	)
	outfile_name = f"{label}_scored"
	result.to_csv(f"{out_path}/{outfile_name}.tsv", index=False, sep="\t")
	print(f"Results saved to {out_path}/{outfile_name}.tsv")

	print("\n\n-----------------------------------Summary of results by layer:--------------------------------------")
	for layer_index, layer_df in result.groupby("layer"):
		layer_accuracy = layer_df["choice1_logprob_greater"].mean() * 100
		print(f"Layer {layer_index}: choice '1' has higher logprob in {layer_accuracy:.2f}% of pairs")


def read_blimp(file_path):
	return pd.read_json(file_path, lines=True)[['sentence_good', 'sentence_bad', 'field', 'linguistics_term']]


def main() -> None:
	parser = argparse.ArgumentParser(
		description="Score BLiMP/EWoK pairs via a forced 1-or-2 multiple-choice prompt, for base or instruction-tuned models."
	)
	parser.add_argument("--data", required=True, help="Dataset key to score")
	parser.add_argument("--model", required=True, choices=list(BASE_MODELS.keys()) + list(INSTRUCT_MODELS.keys()), help="Model name")
	parser.add_argument("--test", required=False, action="store_true", help="Use only the first 10 examples")
	parser.add_argument("--ewok-root", required=False, help="Optional root path to EWoK output dataset files")
	args = parser.parse_args()

	dtype = DATA_TYPE.get(args.data, "blimp")
	is_instruct = args.model in INSTRUCT_MODELS
	if is_instruct:
		model_id = INSTRUCT_MODELS[args.model]
	elif args.model in BASE_MODELS:
		model_id = BASE_MODELS[args.model]
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
			score_dataset(model, tokenizer, data, out_path, f"ewok-{suite_name}", is_instruct)

		return

	data_path = Path(DATA_PATH) / DATA_MAP[args.data]
	data = read_blimp(data_path)
	if args.test:
		data = data.head(10)

	out_path = Path(TEST_OUTDIR if args.test else BASE_OUTDIR) / args.model / args.data
	os.makedirs(out_path, exist_ok=True)
	score_dataset(model, tokenizer, data, out_path, args.data, is_instruct)


if __name__ == "__main__":
	main()
