# Chapter 6 — Fine-Tuning for Classification

Fine-tuning a pretrained GPT-2 model for binary SMS spam classification.

## Artifacts

- `exercises/fine_tuning_for_classification.ipynb` — complete implementation
- `notebooks/chap6_key_learnings.ipynb` — distilled Chapter 6 concepts and core functions

## Main Flow

```text
prepare labeled data
→ load pretrained GPT-2
→ replace output head
→ fine-tune selected layers
→ evaluate
→ classify new text
```


The chapter itself organizes the workflow into dataset preparation, model setup, and model fine-tuning/usage.
