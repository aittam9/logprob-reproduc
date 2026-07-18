import os
from pathlib import Path
from typing import List, Tuple

import pandas as pd
import torch
import nnsight
from nnsight import LanguageModel


# All access to model internals in this codebase goes through nnsight: model
# loading returns an `nnsight.LanguageModel`, and `layer_logprobs` below reads
# every layer's hidden state via nnsight's tracing API rather than manual
# forward hooks. Passing `remote=True` runs the same trace on NDIF instead of
# on local hardware -- see `configure_ndif`.


def configure_ndif(api_key: str = None) -> None:
	"""Point nnsight at NDIF for remote execution.

	Reads `api_key`, falling back to the NDIF_API_KEY environment variable.
	Request a key at https://login.ndif.us. Only needed when running with
	--remote.
	"""
	api_key = api_key or os.environ.get("NDIF_API_KEY")
	if not api_key:
		raise ValueError(
			"Remote execution requires an NDIF API key: set the NDIF_API_KEY "
			"environment variable, or pass --ndif-api-key. Request one at "
			"https://login.ndif.us."
		)
	nnsight.CONFIG.API.APIKEY = api_key


def load_model(model_name: str, device: str = None, remote: bool = False) -> Tuple[LanguageModel, "PreTrainedTokenizer"]:
	"""Load `model_name` as an nnsight `LanguageModel`.

	With `remote=False` (default), the model is dispatched to `device`
	(auto-detected as cuda if available) the first time it's traced. With
	`remote=True`, weights are never downloaded or dispatched locally: the
	forward pass runs on NDIF instead, so `device` is ignored.
	"""
	if remote:
		model = LanguageModel(model_name)
	else:
		if device is None:
			device = "cuda" if torch.cuda.is_available() else "cpu"
		model = LanguageModel(model_name, device_map=device)

	tokenizer = model.tokenizer
	if tokenizer.pad_token is None:
		tokenizer.pad_token = tokenizer.eos_token

	return model, tokenizer


def layer_logprobs(
	model: LanguageModel,
	prefix_ids: List[int],
	continuation_ids: List[int],
	remote: bool = False,
) -> List[Tuple[float, float]]:
	"""Score `continuation_ids`, conditioned on `prefix_ids`, at every layer of
	`model` via a logit-lens: the model's own output head applied directly to
	each layer's hidden state.

	This mirrors what `output_hidden_states=True` returns for a standard
	transformers forward pass: entries for every decoder layer except the
	last are the raw (pre-final-norm) residual stream, while the last entry
	is already post-final-norm -- so applying the output head uniformly to
	every entry gives an approximate logit lens for early layers and the
	model's real output logits for the last one. nnsight's trace requests
	the same `output_hidden_states=True` from the underlying model and reads
	the result off `model.output.hidden_states`; with `remote=True` this runs
	on NDIF and only the small per-layer scalars below are sent back, not the
	hidden states themselves.

	Returns a list of `(total_logprob, avg_logprob)` tuples, one per layer,
	ordered from the earliest layer to the last.
	"""
	input_ids = torch.tensor([prefix_ids + continuation_ids])
	# The first token of the sequence is never scored (there's no context to
	# predict it from), so even an empty prefix behaves as if it had length 1.
	prefix_len = max(len(prefix_ids), 1)

	layer_scores = []
	with model.trace(input_ids, output_hidden_states=True, remote=remote):
		hidden_states = model.output.hidden_states
		for hidden_state in hidden_states[1:]:
			logits = model.lm_head(hidden_state)
			shift_logits = logits[:, :-1, :]
			shift_labels = input_ids[:, 1:]

			cont_shift_logits = shift_logits[:, prefix_len - 1 :, :]
			cont_shift_labels = shift_labels[:, prefix_len - 1 :]

			token_logprobs = torch.log_softmax(cont_shift_logits, dim=-1).gather(
				2, cont_shift_labels.unsqueeze(-1)
			).squeeze(-1)

			layer_scores.append((token_logprobs.sum().save(), token_logprobs.mean().save()))

	return [(total.item(), avg.item()) for total, avg in layer_scores]


def read_blimp(file_path):
    return pd.read_json(file_path, lines=True)[['sentence_good', 'sentence_bad','field', 'linguistics_term']]


def read_ewok(file_path):
	"""Read an EWoK testsuite file and return a DataFrame with columns:
	`sentence_good`, `sentence_bad`, `field`, `linguistics_term`.

	Supports EWoK schemas with Target1/Target2 and optional Context1/Context2,
	and falls back to simpler pair formats when available.
	"""
	path = Path(file_path)
	if not path.exists():
		raise FileNotFoundError(f"EWoK file not found: {file_path}")

	if path.suffix.lower() == ".csv":
		df = pd.read_csv(path)
	else:
		# assume JSONL
		df = pd.read_json(path, lines=True)

	# If already in expected format
	if {'sentence_good', 'sentence_bad'}.issubset(df.columns):
		for col in ['field', 'linguistics_term']:
			if col not in df.columns:
				df[col] = None
		return df[['sentence_good', 'sentence_bad', 'field', 'linguistics_term']]

	def make_sentence(context, target):
		context = '' if pd.isna(context) else str(context)
		target = '' if pd.isna(target) else str(target)
		if '{}' in context:
			return context.replace('{}', target).strip()
		if target and target in context:
			return context.strip()
		return f"{context} {target}".strip()

	# EWoK schema support: use Target1/Target2 and optional Context1/Context2
	if {'Target1', 'Target2'}.issubset(df.columns):
		if {'Context1', 'Context2'}.issubset(df.columns):
			good = df.apply(lambda row: make_sentence(row['Context1'], row['Target1']), axis=1)
			bad = df.apply(lambda row: make_sentence(row['Context2'], row['Target2']), axis=1)
		else:
			good = df['Target1'].astype(str)
			bad = df['Target2'].astype(str)

		field_col = df['Domain'] if 'Domain' in df.columns else pd.Series([None] * len(df))
		if 'TemplateName' in df.columns:
			ling_col = df['TemplateName']
		elif 'ItemTags' in df.columns:
			ling_col = df['ItemTags']
		elif {'ConceptA', 'ConceptB'}.issubset(df.columns):
			ling_col = df['ConceptA'].astype(str) + '/' + df['ConceptB'].astype(str)
		else:
			ling_col = pd.Series([None] * len(df))

		return pd.DataFrame({
			'sentence_good': good,
			'sentence_bad': bad,
			'field': field_col,
			'linguistics_term': ling_col,
		})

	# Try common alternate pair column names
	candidate_pairs = [
		('correct', 'incorrect'),
		('target', 'distractor'),
		('gold', 'alt'),
		('answer', 'other'),
		('good', 'bad'),
		('choiceA', 'choiceB'),
		('choice_1', 'choice_2'),
		('sentence_A', 'sentence_B'),
	]

	for a, b in candidate_pairs:
		if a in df.columns and b in df.columns:
			field_col = df['field'] if 'field' in df.columns else pd.Series([None] * len(df))
			ling_col = df['linguistics_term'] if 'linguistics_term' in df.columns else pd.Series([None] * len(df))
			return pd.DataFrame({
				'sentence_good': df[a].astype(str),
				'sentence_bad': df[b].astype(str),
				'field': field_col,
				'linguistics_term': ling_col,
			})

	# Fallback: take the first two object/text columns
	text_cols = [c for c in df.columns if df[c].dtype == object]
	if len(text_cols) >= 2:
		a, b = text_cols[0], text_cols[1]
		field_col = df['field'] if 'field' in df.columns else pd.Series([None] * len(df))
		ling_col = df['linguistics_term'] if 'linguistics_term' in df.columns else pd.Series([None] * len(df))
		return pd.DataFrame({
			'sentence_good': df[a].astype(str),
			'sentence_bad': df[b].astype(str),
			'field': field_col,
			'linguistics_term': ling_col,
		})

	raise ValueError(f"Could not interpret EWoK file columns: {list(df.columns)}")
