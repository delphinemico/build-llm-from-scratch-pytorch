# Glossary

Book-wide definitions and terminology (in Alphabetical order).

## Entries

### Attention mechanism

A mechanism that allows a model to weigh the relevance of different tokens when processing or generating text.

### Autoregressive model

A model that generates a sequence one token at a time, using previously generated tokens to predict the next token.

### Base model

A pretrained model that has learned general patterns from a large dataset but has not yet been adapted to a specific task. Also called a **foundation model**.

### Classification fine-tuning

Adapting a pretrained model using examples paired with class labels so that it can perform a classification task.

### Context window

The maximum number of tokens a model can consider at one time.

### Decoder-only model

A transformer model that uses only decoder-style components and generates text autoregressively. GPT is a decoder-only architecture.

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

### Instruction fine-tuning

Adapting a pretrained model using instruction–response examples so that it learns to follow user requests.

### Large language model (LLM)

A deep neural network trained on large amounts of text to model, process, and generate language.

### Next-token prediction

The training task of predicting the token that follows a given sequence of previous tokens.

### Parameter

A trainable value, such as a weight or bias, that is updated during model training.

### Pretraining

The initial large-scale training stage in which a model learns general language patterns from raw, unlabeled text.

### Self-supervised learning

A learning approach in which labels are generated from the structure of the input data itself. In LLM pretraining, the next token serves as the label.

### Token

A unit of text processed by a language model, such as a word, subword, punctuation mark, or character sequence.

### Tokenization

The process of splitting raw text into tokens that can be converted into numerical token IDs.

### Transformer

A neural network architecture built around attention mechanisms for processing sequences and modeling relationships between tokens.

### Zero-shot learning

Performing a task without task-specific examples or additional parameter updates.