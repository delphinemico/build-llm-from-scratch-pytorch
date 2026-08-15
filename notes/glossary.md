# Glossary

Book-wide definitions and terminology (in alphabetical order).

## Entries

### Attention head

One independent set of query, key, and value projections within multi-head attention.

### Attention mechanism

A mechanism that allows a model to weigh the relevance of different tokens when processing or generating text.

### Attention score

An unnormalized similarity value calculated between a query and a key.

### Attention weight

A normalized attention score that determines how strongly one token attends to another token.

### Autoregressive generation

Generating a sequence one token at a time using previously generated tokens as context.

### Autoregressive model

A model that predicts or generates the next token using previously observed tokens as context.

### Backpropagation

The process of computing gradients of the loss with respect to model parameters.

### Base model

A pretrained model that has learned general patterns from a large dataset but has not yet been adapted to a specific downstream task.

### Byte pair encoding (BPE)

A subword tokenization method that builds a vocabulary from frequently occurring character and subword units.

### Categorical sampling

Selecting one token according to a probability distribution over possible next tokens. In PyTorch, this can be performed with `torch.multinomial`.

### Causal attention

Self-attention in which each token can attend only to itself and previous tokens.

### Causal mask

A mask that prevents tokens from attending to future positions during autoregressive language modeling.

### Checkpoint

A saved model state containing model parameters and, optionally, optimizer state so that inference or training can later be resumed.

### Classification fine-tuning

Adapting a pretrained model using labeled examples so that it can perform a classification task.

### Classification head

A final model layer that maps learned representations to logits for the target classes.

### Class imbalance

A dataset condition in which some classes contain substantially more examples than others.

### Class logit

A raw output score assigned by the model to a possible class before normalization.

### Context vector

A weighted combination of value vectors that represents a token in relation to the surrounding sequence.

### Context window

The maximum number of tokens a model can process as one sequence.

### Cross-entropy loss

A loss function that measures how well predicted class or token distributions match the correct targets.

### Custom collate function

A function used by a DataLoader to construct batches, for example by padding variable-length sequences and preparing input–target pairs.

### DataLoader

A PyTorch utility that groups dataset examples into batches and optionally handles shuffling and parallel data loading.

### Decoder-only model

A transformer architecture that uses causal self-attention and generates text autoregressively. GPT is a decoder-only model.

### Distributed Data Parallel (DDP)

A PyTorch approach to multi-GPU training in which each process holds a model replica and gradients are synchronized across processes.

### DistributedSampler

A PyTorch sampler that partitions a dataset so that different DDP processes receive different subsets of the training data.

### Embedding

A continuous numerical vector representation of a discrete object such as a token.

### Emergent behavior

A capability that arises from large-scale training even though the model was not explicitly trained for that specific behavior.

### Feed-forward network

A neural network applied independently to each token representation inside a transformer block.

### Few-shot learning

Performing a task using a small number of examples provided in the prompt without updating model parameters.

### Fine-tuning

Additional training that adapts a pretrained model to a particular task, domain, or behavior.

### Foundation model

A model pretrained on broad data that can later be adapted to many downstream tasks.

### Gradient

The derivative of the loss with respect to a model parameter, indicating how that parameter should change to reduce the loss.

### Greedy decoding

A generation strategy that selects the highest-scoring token at every generation step.

### Head dimension

The portion of the attention output dimension assigned to each attention head:

`head_dim = d_out / num_heads`.

### Ignore index (`-100`)

A target value ignored by PyTorch cross-entropy loss, commonly used to prevent padded positions from contributing to the training loss.

### In-context learning (ICL)

Performing a task from instructions or examples provided in the prompt without updating model parameters.

### Inference

Using a trained model to generate outputs or make predictions without updating its parameters.

### Input–target pair

A pair of sequences used for next-token prediction where the target sequence is the input sequence shifted one token forward.

### Instruction dataset

A dataset containing instructions, optional inputs, and expected responses used for supervised instruction fine-tuning.

### Instruction fine-tuning

Adapting a pretrained language model using instruction–response examples so that it learns to follow natural-language instructions.

### Key

A learned representation against which a query is compared when calculating attention scores.

### Large language model (LLM)

A neural network trained on large amounts of text to model, process, and generate language.

### Last-token representation

The contextual representation at the final sequence position. In a causal transformer, it can incorporate information from all preceding tokens.

### Layer normalization

A normalization operation applied across the embedding dimension of each token representation.

### Logit

A raw model output score produced before applying softmax.

### Multi-head attention

An attention mechanism that performs several attention operations in parallel so that different heads can learn different token relationships.

### Next-token prediction

The training objective of predicting the token that follows a sequence of previous tokens.

### Optimizer

An algorithm that updates model parameters using their gradients.

### Optimizer state

Internal values maintained by an optimizer, such as AdamW's running moment estimates, that are required to faithfully resume training.

### Output head

The final linear layer that maps model representations to task-specific output logits, such as vocabulary logits or class logits.

### Output projection

A learned linear transformation applied after combining the outputs of multiple attention heads.

### Padding

Adding special tokens to shorter sequences so that examples in the same batch have a common length.

### Parameter

A trainable model value, such as a weight or bias, that is updated during training.

### Positional embedding

A vector representation that encodes a token's position within a sequence.

### Pretrained weights

Model parameters learned during a previous training stage and loaded into a compatible model architecture.

### Pretraining

The initial large-scale training stage in which a model learns general language patterns from raw text, typically through next-token prediction.

### Prompt template

A consistent textual structure used to format instructions, optional inputs, and expected responses.

### Query

A learned representation used to determine which tokens are relevant to the current token.

### Rank

The unique identifier of a process participating in distributed training.

### Residual connection

A connection that adds a sublayer's input directly to its transformed output, helping preserve information and gradient flow. Also called a shortcut or skip connection.

### Scaled dot-product attention

Attention that computes query–key dot products, scales them by the square root of the key dimension, and normalizes them with softmax.

### Self-attention

An attention mechanism in which queries, keys, and values are derived from the same input sequence.

### Self-supervised learning

A learning approach in which training targets are derived from the structure of the input data itself. In GPT pretraining, the next token serves as the target.

### Sliding window

A method for creating training samples by moving a fixed-length window across tokenized text.

### Special token

A token reserved for a particular purpose, such as padding, marking boundaries, or representing unknown text.

### Stride

The number of token positions by which a sliding window moves when creating the next training sample.

### Supervised fine-tuning (SFT)

Further training of a pretrained model using labeled input–output examples, such as instruction–response pairs.

### Temperature

A generation parameter that controls the sharpness of the next-token probability distribution. Lower values make generation more deterministic; higher values increase diversity.

### Token

A unit of text processed by a language model, such as a word, subword, punctuation mark, or character sequence.

### Token embedding

A trainable vector representation associated with a token ID.

### Token ID

The integer assigned to a token in a tokenizer's vocabulary.

### Tokenization

The process of converting raw text into tokens that can be mapped to numerical token IDs.

### Top-K sampling

A generation strategy that restricts candidate next tokens to the `K` highest-scoring tokens before sampling.

### Transformer

A neural network architecture built around attention mechanisms for processing sequences and modeling relationships between tokens.

### Transformer block

A repeated GPT building block combining causal multi-head attention, feed-forward processing, normalization, and residual connections.

### Undersampling

Reducing the number of examples from a majority class to create a more balanced dataset.

### Value

The learned token information that is combined according to the attention weights.

### Vocabulary

The complete set of tokens recognized by a tokenizer together with their corresponding token IDs.

### Zero-shot learning

Performing a task without task-specific examples in the prompt or additional parameter updates.