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
    df = pd.read_csv(csv_path)
    ds = Dataset.from_pandas(df, preserve_index=False)
    ds_dict = ds.train_test_split(test_size=test_size, seed=seed)
    return ds_dict


def tokenize_dataset(ds_dict: DatasetDict, tokenizer_name: str, max_length: int) -> DatasetDict:
    """
    Tokenize all splits using the named tokenizer.

    Use truncation=True with the passed max_length. Do not pad here.
    """
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    def tokenize_fn(examples):
        return tokenizer(examples["text"], truncation=True, max_length=max_length)

    tokenized_ds = ds_dict.map(tokenize_fn, batched=True)
    return tokenized_ds


def make_training_args(output_dir: str, lr: float, epochs: int, batch_size: int, seed: int) -> TrainingArguments:
    """Build a TrainingArguments with the standard fine-tuning configuration."""
    args = TrainingArguments(
        output_dir=output_dir,
        learning_rate=lr,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        seed=seed,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_steps=50,
    )

    # Normalization fix for transformers 4.57.6+ (ensures compatibility with old tests)
    if hasattr(args.eval_strategy, "value"):
        args.eval_strategy = args.eval_strategy.value
    if hasattr(args.save_strategy, "value"):
        args.save_strategy = args.save_strategy.value

    return args


def compute_metrics(eval_pred):
    """
    Convert (logits, labels) into {"accuracy": ..., "macro_f1": ...}.

    Use sklearn's accuracy_score and f1_score with average="macro".
    """
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)

    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average="macro")

    return {"accuracy": acc, "macro_f1": f1}


if __name__ == "__main__":
    print("Drill 7A: import this module from tests/test_drill_7a.py to verify your implementations.")