# Chapter 2 — Working with Text Data

This chapter implements the text-processing pipeline that converts raw text
into numerical input representations for a GPT-like language model.

## Artifacts

- [`notebooks/chap2_key_learnings.ipynb`](notebooks/chap2_key_learnings.ipynb)  
  Concise conceptual notes, definitions, mental models, and tensor shapes.

- [`exercises/working_with_text_data.ipynb`](exercises/working_with_text_data.ipynb)  
  Full implementation of tokenization, sliding-window sampling, batching,
  token embeddings, and positional embeddings.

## Main pipeline

```text
Raw text  
→ tokens  
→ token IDs  
→ input/target sequences  
→ batches  
→ token embeddings  
→ token + positional embeddings  
→ transformer input
```  