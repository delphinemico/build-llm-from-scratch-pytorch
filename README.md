# Building an LLM from Scratch with PyTorch

This is my personal learning repository for Sebastian Raschka's
*Build a Large Language Model (From Scratch)*.

It contains selected notebooks, runnable scripts, exercises, and notes that I
found valuable enough to preserve. The goal is to document my learning without
duplicating the book or creating unnecessary maintenance work.

## Current status

| Section | Status |
|---|---|
| Appendix A — PyTorch foundations | Artifacts in progress |
| Chapter 1 | Not started |

See [`notes/progress.md`](notes/progress.md) for the detailed study log.

## Repository organization

- `appendix_A/`: selected PyTorch notes, scripts, and exercises
- `chapter_01/`: artifacts created while studying Chapter 1
- `notes/`: book-wide glossary, PyTorch patterns, and progress

## Local environment

Activate the dedicated environment in Git Bash:

```bash
source ~/venvs/llmbookvenv/Scripts/activate
```

The local environment is CPU-only. Single-GPU work will run in Google Colab,
and multi-GPU DDP work will run on RunPod.