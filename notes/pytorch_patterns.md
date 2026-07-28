# PyTorch Patterns

Reusable PyTorch patterns encountered throughout the book.

## Training mode

```python
model.train()
```

Training mode enables training-specific behavior in layers such as dropout and batch normalization.

## Evaluation mode

Use evaluation mode before validation, testing, or inference:

```python
model.eval()
```

Evaluation mode:

- disables dropout,
- makes batch normalization use its stored running statistics,
- does not, by itself, disable gradient tracking.

## Disable gradient tracking

Use `torch.no_grad()` when gradients are not needed:

```python
with torch.no_grad():
    predictions = model(features)
```

This reduces memory usage and avoids constructing the computation graph during evaluation or inference.

A typical evaluation pattern is:

```python
model.eval()

with torch.no_grad():
    for features, targets in data_loader:
        predictions = model(features)
```

## Clear accumulated gradients

```python
optimizer.zero_grad()
```

PyTorch accumulates gradients in parameter `.grad` attributes by default, so gradients should normally be cleared before each backward pass.

## Backpropagation

```python
loss.backward()
```

This computes gradients of the loss with respect to trainable model parameters.

## Parameter update

```python
optimizer.step()
```

This updates the model parameters using the gradients computed during backpropagation.