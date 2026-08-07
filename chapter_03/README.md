# Chapter 3 — Coding Attention Mechanisms

This chapter implements the attention mechanism used in GPT-like language models.

## Artifacts

- [`notebooks/chap3_key_learnings.ipynb`](notebooks/chap3_key_learnings.ipynb)  
  Concise mental models, tensor shapes, definitions, final attention classes,   and concept checks.

- [`exercises/coding_attention_mechanisms.ipynb`](exercises/coding_attention_mechanisms.ipynb)  
  Detailed implementation of self-attention, causal attention, and multi-head   attention.

## Main Progression

```text
Simplified self-attention
→ trainable Q, K, and V projections
→ scaled dot-product attention
→ causal masking
→ attention dropout
→ multi-head attention
```

## Main Output

Attention transforms token embeddings into context-aware representations:  

```text
Input:  [batch_size, num_tokens, embedding_dim]
Output: [batch_size, num_tokens, embedding_dim]
```