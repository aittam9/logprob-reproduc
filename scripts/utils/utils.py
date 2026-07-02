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
def load_model(model_name: str, device:str | None =  None, cache_dir: str | None = None):
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
