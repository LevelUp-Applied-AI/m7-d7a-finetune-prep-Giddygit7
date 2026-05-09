"""Drill 7A autograder."""
import os

import numpy as np
import pytest

# sys.path is set by starter/conftest.py — no insertion here.

FIXTURE = os.path.join(os.path.dirname(__file__), "..", "fixtures", "tiny_app_reviews.csv")


def test_make_dataset_returns_dict_with_train_and_test():
    import drill
    ds = drill.make_dataset(FIXTURE, test_size=0.25, seed=42)
    assert "train" in ds and "test" in ds


def test_make_dataset_split_proportions():
    import drill
    ds = drill.make_dataset(FIXTURE, test_size=0.25, seed=42)
    total = len(ds["train"]) + len(ds["test"])
    assert abs(len(ds["test"]) - total * 0.25) <= 1


def test_tokenize_dataset_has_input_ids_and_mask():
    import drill
    ds = drill.make_dataset(FIXTURE, test_size=0.25, seed=42)
    tokenized = drill.tokenize_dataset(ds, "distilbert-base-uncased", max_length=32)
    for split in ("train", "test"):
        cols = tokenized[split].column_names
        assert "input_ids" in cols and "attention_mask" in cols


def test_tokenize_dataset_max_length_truncates():
    import drill
    ds = drill.make_dataset(FIXTURE, test_size=0.25, seed=42)
    tokenized = drill.tokenize_dataset(ds, "distilbert-base-uncased", max_length=8)
    max_seen = max(len(x) for x in tokenized["train"]["input_ids"])
    assert max_seen <= 8


def test_make_training_args_attributes():
    import drill
    args = drill.make_training_args("model", lr=2e-5, epochs=3, batch_size=16, seed=99)
    assert args.learning_rate == 2e-5
    assert args.num_train_epochs == 3
    assert args.per_device_train_batch_size == 16
    assert args.seed == 99
    # Per the drill guide, set evaluation/save cadence to once per epoch and
    # logging cadence to every ~50 steps. The eval_strategy attribute is named
    # eval_strategy (not evaluation_strategy) in transformers>=4.41 — the
    # course pins that range in requirements.txt.
    assert str(args.eval_strategy) == "epoch", \
        f"eval_strategy must be 'epoch' (got {args.eval_strategy!r})"
    assert str(args.save_strategy) == "epoch", \
        f"save_strategy must be 'epoch' (got {args.save_strategy!r})"
    assert args.logging_steps == 50, \
        f"logging_steps must be 50 (got {args.logging_steps})"


def test_compute_metrics_returns_accuracy_and_macro_f1():
    import drill
    logits = np.array([[0.1, 0.7, 0.2], [0.6, 0.2, 0.2]])
    labels = np.array([1, 0])
    result = drill.compute_metrics((logits, labels))
    assert "accuracy" in result and "macro_f1" in result


def test_compute_metrics_correct_on_perfect_predictions():
    import drill
    logits = np.array([[0.1, 0.9, 0.0], [0.9, 0.05, 0.05], [0.0, 0.1, 0.9]])
    labels = np.array([1, 0, 2])
    r = drill.compute_metrics((logits, labels))
    assert abs(r["accuracy"] - 1.0) < 1e-9
    assert abs(r["macro_f1"] - 1.0) < 1e-9


def test_compute_metrics_correct_on_known_confusion():
    """Catches average='weighted' instead of 'macro'."""
    import drill
    # 4 samples, 2 correct → accuracy 0.5; macro-F1 = 0.5
    logits = np.array([[0.9, 0.1], [0.1, 0.9], [0.9, 0.1], [0.1, 0.9]])
    labels = np.array([0, 0, 1, 1])
    r = drill.compute_metrics((logits, labels))
    assert abs(r["accuracy"] - 0.5) < 1e-9
    assert abs(r["macro_f1"] - 0.5) < 1e-9
