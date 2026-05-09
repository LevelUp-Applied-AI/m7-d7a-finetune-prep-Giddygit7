"""
Module 7 Week A — Drill: Fine-Tuning Prep.

Implement the four TODO functions. The drill does not run training — that is
tomorrow's lab. The drill exercises the mechanical preparation steps.
"""

import numpy as np
import pandas as pd
from datasets import Dataset, DatasetDict
from sklearn.metrics import accuracy_score, f1_score
from transformers import AutoTokenizer, TrainingArguments


def make_dataset(csv_path: str, test_size: float, seed: int) -> DatasetDict:
    """
    Load a CSV with `text` and `label` columns; split into train/test.

    Returns a DatasetDict with keys "train" and "test".
    """
    # TODO: read csv_path with pandas
    # TODO: convert to a Hugging Face Dataset (preserve_index=False)
    # TODO: split with the passed test_size and seed
    # TODO: return the resulting DatasetDict
    raise NotImplementedError


def tokenize_dataset(ds_dict: DatasetDict, tokenizer_name: str, max_length: int) -> DatasetDict:
    """
    Tokenize all splits using the named tokenizer.

    Use truncation=True with the passed max_length. Do not pad here.
    """
    # TODO: load tokenizer with AutoTokenizer.from_pretrained
    # TODO: define a tokenize_fn that calls the tokenizer with truncation + max_length
    # TODO: apply ds_dict.map with batched=True
    # TODO: return the tokenized DatasetDict
    raise NotImplementedError


def make_training_args(output_dir: str, lr: float, epochs: int, batch_size: int, seed: int) -> TrainingArguments:
    """Build a TrainingArguments with the standard fine-tuning configuration."""
    # TODO: return a TrainingArguments configured with the passed arguments.
    # In addition to wiring the kwargs through, set:
    #   - eval_strategy="epoch"           (renamed from evaluation_strategy in transformers 4.41+)
    #   - save_strategy="epoch"
    #   - logging_steps=50
    # The course pins transformers>=4.41,<5.0 — use the new argument names.
    raise NotImplementedError


def compute_metrics(eval_pred):
    """
    Convert (logits, labels) into {"accuracy": ..., "macro_f1": ...}.

    Use sklearn's accuracy_score and f1_score with average="macro".
    """
    # TODO: unpack eval_pred to logits, labels
    # TODO: argmax logits over axis 1
    # TODO: compute accuracy and macro-F1
    # TODO: return as a dict
    raise NotImplementedError


if __name__ == "__main__":
    print("Drill 7A: import this module from tests/test_drill_7a.py to verify your implementations.")
