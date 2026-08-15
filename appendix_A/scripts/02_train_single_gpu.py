# Copyright (c) Sebastian Raschka under Apache License 2.0 (see LICENSE.txt).
# Source for "Build a Large Language Model From Scratch"
#   - https://www.manning.com/books/build-a-large-language-model-from-scratch
# Code: https://github.com/rasbt/LLMs-from-scratch
#
# Modified and annotated by Delphine Mico, 2026, for educational study.

"""Device-aware PyTorch training loop for a single GPU."""

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from common import NeuralNetwork, create_toy_datasets, compute_accuracy
torch.manual_seed(123)

train_ds, test_ds = create_toy_datasets()
train_loader = DataLoader(dataset=train_ds, batch_size=2, shuffle=True, num_workers=0, drop_last=True) # drop_last=True is OPTIONAL
test_loader = DataLoader(dataset=test_ds, batch_size=2, shuffle=False, num_workers=0)

model = NeuralNetwork(num_inputs=2, num_outputs=2)
if not torch.cuda.is_available():
    raise RuntimeError("This script requires a CUDA-enabled GPU")
device = torch.device("cuda") # addition No1 (from 01_train_cpu.py)
model.to(device) # addition No2 (from 01_train_cpu.py)
optimizer = torch.optim.SGD(model.parameters(), lr=0.5)
num_epochs = 3

for epoch in range(num_epochs):
    model.train()
    for batch_idx, (features, labels) in enumerate(train_loader):
        features, labels = features.to(device), labels.to(device) # addition No3 (from 01_train_cpu.py)
        logits = model(features)
        loss = F.cross_entropy(logits, labels)
        optimizer.zero_grad() # prevents gradients accumulation
        loss.backward() # calculates gradients from computation graph
        optimizer.step() # uses the gradients to update the model parameters as to minimize loss
        # LOGGING
        print(f"Epoch: {epoch+1:03d} | Batch {batch_idx:03d} | Train Loss: {loss:.2f}")
    model.eval()

# compute accuracy
train_acc = compute_accuracy(model, train_loader, device=device)
print(f"Training accuracy: {train_acc:.2%}")
test_acc = compute_accuracy(model, test_loader, device=device)
print(f"Test accuracy: {test_acc:.2%}")