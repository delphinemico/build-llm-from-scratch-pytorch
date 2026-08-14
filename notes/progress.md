# Progress Log

## Current status

| Section | Status | Started | Completed |
|---|---|---|---|
| Appendix A — Introduction to PyTorch | Completed |  | July 27, 2026 |
| Chapter 1 — Understanding Large Language Models | Completed | July 30, 2026 | July 30, 2026 |
| Chapter 2 — Working with Text Data | Completed | July 31, 2026 | August 4, 2026 |
| Chapter 3 — Coding Attention Mechanisms | Completed | August 4, 2026 | August 6, 2026 |
| Chapter 4 — Implementing a GPT model from scratch to generate text | Completed | August 6, 2026 | August 7, 2026 |
| Chapter 5 — Pretraining on unlabeled data | Completed | August 10, 2026 | August 12, 2026 |
| Chapter 6 — Fine-Tuning for Classification | Completed | August 12, 2026 | August 14, 2026 |

## Study log

### July 27, 2026

- Created and configured the private GitHub repository.
- Established the chapter template and generated `chapter_01`.
- Added the Appendix A artifact structure.
- Configured the external `llmbookvenv` Python environment.
- Installed local CPU PyTorch.
- Configured VS Code and Jupyter to use `llmbookvenv`.
- Recorded exact package versions in `requirements-lock.txt`.

### July 28, 2026

- Reviewed PyTorch Distributed Data Parallel in depth.
- Reconstructed and annotated the DDP training script.
- Simplified the repository so notebooks remain the primary learning artifact.
- Finalized the shared Appendix A utilities.
- Implemented and ran the standard CPU training loop locally.
- Prepared the single-GPU implementation for Google Colab.
- Reconstructed and documented the multi-GPU DDP implementation.
- Appendix A conceptual study is complete.
- Chapter 1 has not yet been started.

### July 30, 2026

- Started and completed Chapter 1, *Understanding Large Language Models*.

### July 31, 2026
- Created the Chapter 1 key-learnings notebook.
- Recorded concise definitions and important conceptual takeaways.
- Documented the three stages of coding an LLM:
  1. building the architecture,
  2. pretraining a foundation model,
  3. fine-tuning for classification or instruction following.
- Summarized the roles of tokenization, context length, attention, pretraining, and fine-tuning.
- Added short answers to questions to revisit in later chapters.
- Started Chapter 2, *Working with text data*.

### August 4, 2026

- Completed Chapter 2, *Working with Text Data*.
- Implemented tokenization and token-to-ID conversion.
- Studied special tokens and byte pair encoding.
- Implemented sliding-window sampling for next-token prediction.
- Created `GPTDatasetV1` and a PyTorch `DataLoader`.
- Converted token IDs into trainable token embeddings.
- Added positional embeddings to encode token order.
- Documented the complete raw-text-to-transformer-input pipeline.
- Started Chapter 3, *Coding Attention Mechanisms*.

### August 5, 2026

- Finished reading Chapter 3, *Coding Attention mechanisms*.

### August 6, 2026

- Completed Chapter 3, *Coding Attention Mechanisms*.
- Implemented simplified self-attention and trainable scaled dot-product attention.
- Studied the roles of queries, keys, values, attention scores, and context vectors.
- Implemented causal masking to prevent future-token information leakage.
- Applied dropout to attention weights for regularization.
- Implemented `CausalAttention` for batched inputs.
- Compared a simple multi-head wrapper with an efficient weight-splitting
  implementation.
- Implemented the final `MultiHeadAttention` module used by the GPT model.
- Documented the complete attention tensor-shape flow.
- Started Chapter 4, *Implementing a GPT model from scratch to generate text*.

### August 7, 2026

- Completed Chapter 4, *Implementing a GPT Model from Scratch to Generate Text*.
- Implemented `LayerNorm`, `FeedForward`, and `TransformerBlock`.
- Assembled the full `GPTModel`.
- Implemented simple autoregressive text generation.
- Documented the main GPT tensor-shape flow.

### August 10, 2026
- Started Chapter 5, *Pretraining on unlabeled data*.

### August 11, 2026
- Continued Chapter 5, *Pretraining on unlabeled data*.

### August 12, 2026

- Completed Chapter 5, *Pretraining on Unlabeled Data*.
- Implemented next-token loss calculation and the LLM training loop.
- Studied greedy, temperature, multinomial, and Top-K decoding.
- Implemented the final configurable `generate()` function.
- Saved and restored model and optimizer states.
- Loaded pretrained OpenAI GPT-2 weights into the custom `GPTModel`.

### August 13, 2026
- Finished reading Chapter 6, *Fine-Tuning for Classification*.

### August 14, 2026

- Completed Chapter 6, *Fine-Tuning for Classification*.
- Prepared and balanced the SMS spam dataset.
- Implemented the `SpamDataset` classification dataset.
- Adapted pretrained GPT-2 with a binary classification head.
- Implemented classification loss, accuracy, and training utilities.
- Fine-tuned and evaluated the GPT-based spam classifier.
- Used the fine-tuned model to classify new text.