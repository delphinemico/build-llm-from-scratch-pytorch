# Building a Large Language Model from Scratch with PyTorch

This repository documents my hands-on implementation and study of Sebastian Raschka's *Build a Large Language Model (From Scratch)*.

The goal of this project was to deeply understand the mechanics of GPT-style large language models by implementing the major components directly in PyTorch, progressing from text preprocessing and attention mechanisms through GPT architecture, pretraining, classification fine-tuning, and instruction fine-tuning.

Rather than reproducing the book, this repository preserves selected implementations, exercises, distilled key learnings, diagrams, and notes that I found most useful for understanding how the pieces fit together.

## What This Repository Covers

The project follows the full progression from raw text to a fine-tuned instruction-following GPT model:

```text
Raw text
→ tokenization
→ token IDs
→ embeddings
→ self-attention
→ causal multi-head attention
→ transformer blocks
→ GPT architecture
→ next-token pretraining
→ text generation
→ classification fine-tuning
→ instruction fine-tuning
```

## Repository Structure

```text
build-llm-from-scratch-pytorch/
│
├── appendix_A/
│   ├── notebooks/
│   └── scripts/
│
├── chapter_01/
│   └── notebooks/
│
├── chapter_02/
│   ├── exercises/
│   ├── notebooks/
│   └── scripts/
│
├── chapter_03/
│   ├── exercises/
│   ├── notebooks/
│   └── scripts/
│
├── chapter_04/
│   ├── notebooks/
│   └── scripts/
│
├── chapter_05/
│   ├── exercises/
│   ├── notebooks/
│   └── scripts/
│
├── chapter_06/
│   ├── exercises/
│   └── notebooks/
│
├── chapter_07/
│   ├── exercises/
│   └── notebooks/
│
├── notes/
│   ├── glossary.md
│   └── progress.md
│
├── .gitignore
├── requirements.txt
├── ATTRIBUTION.md
├── LICENSE.txt
└── README.md
```

Each chapter generally contains:

- **`notebooks/`** — distilled key learnings and core implementation concepts
- **`exercises/`** — more complete hands-on implementations and chapter exercises
- **`scripts/`** — reusable implementations extracted from the notebooks where appropriate

## Chapter Overview

| Section | Main Topics |
|---|---|
| **Appendix A** | PyTorch training loops, gradients, optimization, single-GPU training, Distributed Data Parallel |
| **Chapter 1** | LLM concepts, autoregressive modeling, pretraining, fine-tuning |
| **Chapter 2** | Tokenization, token IDs, sliding-window datasets, embeddings |
| **Chapter 3** | Self-attention, Q/K/V projections, causal masking, multi-head attention |
| **Chapter 4** | Layer normalization, feed-forward networks, transformer blocks, GPT architecture |
| **Chapter 5** | Next-token pretraining, loss computation, text generation, sampling, checkpoints, pretrained GPT-2 weights |
| **Chapter 6** | GPT classification fine-tuning, dataset balancing, classification head, spam classification |
| **Chapter 7** | Instruction datasets, prompt formatting, custom batching, supervised instruction fine-tuning |

## Core GPT Architecture

The model implemented in this repository follows the decoder-only GPT architecture:

```text
Token IDs
    ↓
Token Embeddings
    +
Positional Embeddings
    ↓
Transformer Blocks
    │
    ├── LayerNorm
    ├── Causal Multi-Head Attention
    ├── Residual Connection
    ├── LayerNorm
    ├── Feed-Forward Network
    └── Residual Connection
    ↓
Final LayerNorm
    ↓
Output Head
    ↓
Vocabulary Logits
    ↓
Next-Token Prediction
```

## Attention

A major part of the project is understanding attention from first principles.

The progression implemented in Chapter 3 is:

```text
input embeddings
→ simplified self-attention
→ query, key, and value projections
→ scaled dot-product attention
→ causal masking
→ dropout
→ multi-head attention
→ contextual token representations
```

For multi-head attention, the main tensor flow is:

```text
[b, n, d_in]
→ [b, n, d_out]
→ [b, n, num_heads, head_dim]
→ [b, num_heads, n, head_dim]
→ attention scores [b, num_heads, n, n]
→ context vectors [b, num_heads, n, head_dim]
→ [b, n, d_out]
→ output projection
```

## Pretraining and Text Generation

Chapter 5 trains the GPT model using autoregressive next-token prediction.

```text
Tokenized training data
→ GPT forward pass
→ vocabulary logits
→ next-token cross-entropy loss
→ backpropagation
→ parameter updates
→ evaluation
→ text generation
```

Generation supports both deterministic and probabilistic decoding:

```text
logits
→ optional Top-K filtering
→ temperature scaling
→ softmax
→ greedy selection or categorical sampling
→ next token
```

The chapter also covers model checkpointing and loading pretrained GPT-2 weights into the implemented architecture.

## Classification Fine-Tuning

Chapter 6 adapts the pretrained GPT model for binary text classification.

```text
Raw SMS dataset
→ balance classes
→ train / validation / test split
→ tokenize and batch
→ pretrained GPT
→ replace vocabulary output head with 2-class head
→ fine-tune selected parameters
→ last-token logits
→ cross-entropy loss
→ spam / not spam
```

The GPT architecture remains largely unchanged; the main adaptation is replacing the language-model output head with a classification head and training the model on labeled examples.

## Instruction Fine-Tuning

Chapter 7 extends the pretrained GPT model into an instruction-following model using supervised fine-tuning.

```text
Instruction dataset
→ format instruction + optional input + response
→ tokenize examples
→ pad and batch
→ create shifted input–target pairs
→ ignore padded targets in the loss
→ fine-tune pretrained GPT
→ generate responses
→ evaluate outputs
```

An important takeaway is that the core autoregressive objective remains the same:

```text
Pretraining:
raw text → next-token prediction

Instruction fine-tuning:
instruction + input + response → next-token prediction
```

What changes is primarily the structure and semantics of the training data.

## Key Learning Notebooks

Each chapter includes a compact `key_learnings` notebook designed as a concise technical review of the most important concepts.

These notebooks focus on:

- core mental models
- important tensor shapes
- key classes and functions
- end-to-end data and model flows
- key definitions
- conceptual Q/As

More extensive implementations remain in the corresponding exercise notebooks.

## Book-Wide Glossary

[`notes/glossary.md`](notes/glossary.md) contains a concise alphabetical glossary covering the major concepts used throughout the repository, including:

- attention
- autoregressive generation
- embeddings
- causal masking
- multi-head attention
- transformer blocks
- pretraining
- decoding strategies
- classification fine-tuning
- instruction fine-tuning
- distributed training

## Environment

The project was developed primarily with:

- Python
- PyTorch
- Jupyter
- NumPy
- pandas
- Matplotlib
- tiktoken

Additional dependencies are listed in [`requirements.txt`](requirements.txt).

Local experiments were primarily run on CPU, with GPU-specific experiments performed separately where appropriate.

## Installation

Clone the repository:

```bash
git clone https://github.com/delphinemico/build-llm-from-scratch-pytorch.git
cd build-llm-from-scratch-pytorch
```

Create and activate a virtual environment, then install the dependencies:

```bash
pip install -r requirements.txt
```

Launch Jupyter:

```bash
jupyter lab
```

## Attribution

This repository was created while studying:

**Sebastian Raschka. _Build a Large Language Model (From Scratch)_. Manning, 2024.**

Official accompanying source-code repository:

https://github.com/rasbt/LLMs-from-scratch

Portions of the implementation in this repository follow or adapt the Apache-2.0-licensed source code accompanying the book.

Those files and notebooks contain explicit attribution notices where appropriate.

The organization, annotations, experiments, explanatory notes, diagrams, debugging work, and additional study material in this repository were created as part of my own learning and implementation process.

See [`ATTRIBUTION.md`](ATTRIBUTION.md) for additional details.

## License

See [`LICENSE.txt`](LICENSE.txt).

## Project Status

**Completed.**

The full book implementation and study progression through Chapter 7 has been completed, including:

- PyTorch fundamentals and distributed-training experiments
- tokenizer and embedding pipeline
- attention mechanisms
- GPT architecture implementation
- language-model pretraining
- text generation
- GPT-2 weight loading
- classification fine-tuning
- instruction fine-tuning
- distilled chapter notes and glossary

This repository now serves as a compact reference for the end-to-end mechanics of building and adapting a GPT-style language model from scratch.
