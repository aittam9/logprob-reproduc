import os
import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm.auto import tqdm 


# helper to load models and tokenizers
def load_model(model_name: str, device: str = None, cache_dir: str = None):
	tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
	model = AutoModelForCausalLM.from_pretrained(model_name, cache_dir=cache_dir)

	if tokenizer.pad_token is None:
		tokenizer.pad_token = tokenizer.eos_token

	if device is None:
		device = "cuda" if torch.cuda.is_available() else "cpu"

	model.to(device)
	model.eval()
	return model, tokenizer


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


