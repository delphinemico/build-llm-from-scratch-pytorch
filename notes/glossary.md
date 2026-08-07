# Glossary

Book-wide definitions and terminology (in Alphabetical order).

## Entries

### Attention head

One independent set of query, key, and value projections within multi-head attention.

### Attention mechanism

A mechanism that allows a model to weigh the relevance of different tokens when processing or generating text.

### Attention score

An unnormalized similarity value calculated between a query and a key.

### Attention weight

A normalized attention score that determines how strongly one token attends to another token.

### Autoregressive model

A model that generates a sequence one token at a time, using previously generated tokens to predict the next token.

### Base model

A pretrained model that has learned general patterns from a large dataset but has not yet been adapted to a specific task. Also called a **foundation model**.

### Byte pair encoding

A subword tokenization method that builds a vocabulary by repeatedly merging frequently occurring adjacent units.

### Causal attention

Self-attention that prevents tokens from accessing future positions.

### Causal mask

An upper-triangular mask used to hide future tokens during autoregressive language modeling.

### Classification fine-tuning

Adapting a pretrained model using examples paired with class labels so that it can perform a classification task.

### Context vector

A weighted combination of value vectors that represents a token in relation to the surrounding sequence.

### Context window

The maximum number of tokens a model can process as one sequence.  
In the Chapter 2 data pipeline, this is represented by `max_length`.

### Decoder-only model

A transformer model that uses only decoder-style components and generates text autoregressively. GPT is a decoder-only architecture.

### Embedding

A continuous numerical vector representation of a discrete object such as a token.

### Emergent behavior

A capability that appears as a result of large-scale training even though the model was not explicitly trained for that specific task.

### Fine-tuning

Additional training that adapts a pretrained model to a particular task, domain, or desired behavior.

### Foundation model

A model pretrained on broad data that can later be adapted to many downstream tasks. Also called a **base model**.

### Few-shot learning

Performing a task from a small number of examples provided in the input, without updating the model parameters.

### Gradient

A vector of partial derivatives that indicates how the loss changes with respect to the model parameters.

### Inference

Using a trained model to generate outputs or make predictions without updating its parameters.

### Input–target pair

A pair of sequences used for next-token prediction, where the target sequence is the input sequence shifted one token forward.

### Instruction fine-tuning

Adapting a pretrained model using instruction–response examples so that it learns to follow user requests.

### Key (K projection)

A learned representation against which a query is compared when calculating attention scores.

### Large language model (LLM)

A deep neural network trained on large amounts of text to model, process, and generate language.

### Multi-head attention

An attention mechanism that uses several heads to learn different relationships between tokens.

### Next-token prediction

The training task of predicting the token that follows a given sequence of previous tokens.

### Output projection

A learned linear transformation applied after combining the outputs of multiple attention heads.

### Parameter

A trainable value, such as a weight or bias, that is updated during model training.

### Positional embedding

A vector that represents a token's position within a sequence.

### Pretraining

The initial large-scale training stage in which a model learns general language patterns from raw, unlabeled text.

### Query (Q Projection)

A learned representation used to determine which tokens are relevant to the current token.

### Self-attention

An attention mechanism in which queries, keys, and values are derived from the same input sequence.

### Self-supervised learning

A learning approach in which labels are generated from the structure of the input data itself. In LLM pretraining, the next token serves as the label.

### Scaled dot-product attention

Attention that computes query–key dot products, scales them by the square root of the key dimension, and normalizes them with softmax.

### Sliding window

A method for creating training samples by moving a fixed-length window across tokenized text.

### Special token

A token reserved for a specific purpose, such as marking a document boundary or representing unknown text.

### Stride

The number of token positions by which a sliding window moves when creating the next training sample.

### Token

A unit of text processed by a language model, such as a word, subword, punctuation mark, or character sequence.

### Token embedding

A trainable vector representation associated with a token ID.

### Token ID

The integer assigned to a token in a tokenizer's vocabulary.

### Tokenization

The process of splitting raw text into tokens that can be converted into numerical token IDs.

### Transformer

A neural network architecture built around attention mechanisms for processing sequences and modeling relationships between tokens.

### Value (V Projection)

The learned token information combined according to the attention weights.

### Vocabulary

The complete set of tokens recognized by a tokenizer, together with their corresponding token IDs.

### Zero-shot learning

Performing a task without task-specific examples or additional parameter updates.