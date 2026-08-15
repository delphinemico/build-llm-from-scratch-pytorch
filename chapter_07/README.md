# Chapter 7 — Fine-Tuning to Follow Instructions

Instruction fine-tuning a pretrained GPT model using supervised instruction–response examples.

## Artifacts

- `exercises/fine_tuning_to_follow_instructions.ipynb` — complete implementation
- `notebooks/chap7_key_learnings.ipynb` — distilled Chapter 7 concepts and core code

## Main Flow

```text
instruction data
→ prompt formatting
→ tokenization + batching
→ pretrained GPT
→ supervised fine-tuning
→ response generation
→ evaluation
```