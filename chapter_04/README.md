# Chapter 4 — Implementing a GPT Model

This chapter assembles the components of a GPT-like decoder-only transformer.

## Artifacts

- [`notebooks/chap4_key_learnings.ipynb`](notebooks/chap4_key_learnings.ipynb)  
  Core architecture classes, generation function, and tensor-shape flow.

- [`exercises/implementing_a_gpt_model.ipynb`](exercises/implementing_a_gpt_model.ipynb)  
  Full Chapter 4 implementation and exercises.

## Main Components

```text
Token + positional embeddings
→ TransformerBlock × N
→ Final LayerNorm
→ Output projection
→ Vocabulary logits
→ Autoregressive generation
```