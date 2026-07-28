"""Shared model, dataset, and evaluation utilities for Appendix A."""
import torch
from torch.utils.data import Dataset

class NeuralNetwork(torch.nn.Module):
    """Small multilayer perceptron used in the Appendix A examples."""
    def __init__(self, num_inputs, num_outputs):
        super().__init__()
        self.layers = torch.nn.Sequential(
            # 1st hidden layer
            torch.nn.Linear(num_inputs, 30),
            torch.nn.ReLU(),
            # 2nd hidden layer
            torch.nn.Linear(30, 20),
            torch.nn.ReLU(),
            # output layer
            torch.nn.Linear(20, num_outputs)
        )
    def forward(self, x):
        logits = self.layers(x)
        return logits

class ToyDataset(Dataset):
    """Simple dataset containing feature tensors and labels."""
    def __init__(self, X, y):
        self.features = X
        self.labels = y
    def __getitem__(self, index):
        one_x = self.features[index]
        one_y = self.labels[index]
        return one_x, one_y
    def __len__(self):
        return self.labels.shape[0]

def create_toy_datasets(): # CAREFUL, this function returns Datasets!! NOT DataLoaders
    """Create the shared training and test datasets."""
    # Notice that this creates datasets, not DataLoaders. That is intentional:
        # the CPU script needs ordinary DataLoaders;
        # the DDP script needs a DistributedSampler;
        # therefore, DataLoader creation should remain inside each training script.
    X_train = torch.tensor(
        [
            [-1.2, 3.1],
            [-0.9, 2.9],
            [-0.5, 2.6],
            [2.3, -1.1],
            [2.7, -1.5],
        ]
    )
    y_train = torch.tensor([0, 0, 0, 1, 1])

    X_test = torch.tensor(
        [
            [-0.8, 2.8],
            [2.6, -1.6],
        ]
    )
    y_test = torch.tensor([0, 1])

    train_ds = ToyDataset(X_train, y_train)
    test_ds = ToyDataset(X_test, y_test)
    return train_ds, test_ds

def compute_accuracy(model, dataloader, device="cpu"):
    """Compute classification accuracy without tracking gradients."""
    model.eval()
    correct = 0.0
    total_examples = 0

    for features, labels in dataloader:
        features, labels = features.to(device), labels.to(device)
        with torch.no_grad():
            logits = model(features)
        predictions = torch.argmax(logits, dim=1)
        compare = predictions==labels
        correct += torch.sum(compare)
        total_examples += len(compare)
    return (correct/total_examples).item()
