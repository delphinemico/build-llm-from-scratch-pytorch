# Chapter 5 — Pretraining on Unlabeled Data

## Artifacts

- [`exercises/`](exercises/) — full Chapter 5 implementation and exercises
- [`notebooks/chap5_key_learnings.ipynb`](notebooks/chap5_key_learnings.ipynb) — distilled training and generation concepts
- [`scripts/training.py`](scripts/training.py) — loss, evaluation, and training utilities
- [`scripts/generation.py`](scripts/generation.py) — generation and text-conversion utilities
- [`scripts/gpt_download.py`](scripts/gpt_download.py) — GPT-2 pretrained-weight download utilities
- [`scripts/visualization.py`](scripts/visualization.py) — utilities for plotting training and validation losses

## Main Flow

```text
training data
→ next-token loss
→ pretraining
→ evaluation
→ generation
→ checkpointing / pretrained GPT-2 weights