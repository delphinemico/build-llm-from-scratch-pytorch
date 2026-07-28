"""Multi-GPU training with PyTorch DistributedDataParallel (DDP)."""
# SIMILAR TO DDP-script.py from https://github.com/rasbt/LLMs-from-scratch/tree/main/appendix-A/01_main-chapter-code

# IMPORTS
import os
import platform

import torch
import torch.nn.functional as F
import torch.multiprocessing as mp

from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

# Setting up DDP
def ddp_setup(rank, world_size):
    os.environ("MASTER_ADDR") = "localhost"
    os.environ("MASTER_PORT") = "12345"
    if platform.system() == "Windows":
        os.environ("USE_LIBUV") = "0"
        init_process_group(backend="gloo", rank=rank, world_size=world_size) # "gloo" is Facebook's Collective Communications Library
    else:
        init_process_group(backend="nccl", rank=rank, world_size=world_size) # "nccl" is NVIDIA's Collective Communications Library

# Custom classes
class NeuralNetwork(torch.nn.Module): # simple MLP with 2 Fully connected layers
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
    def __init__(self, X, y):
        self.features = X
        self.labels = y
    def __getitem__(self, index):
        one_x = self.features[index]
        one_y = self.labels[index]
        return one_x, one_y
    def __len__(self):
        return self.labels.shape[0]

# Preparing the dataset
def prepare_dataset():
    # hardcoded train and test data
    X_train = torch.tensor([
        [-1.2, 3.1],
        [-0.9, 2.9],
        [-0.5, 2.6],
        [2.3, -1.1],
        [2.7, -1.5]
    ])
    y_train = torch.tensor([0,0,0,1,1])
    X_test = torch.tensor([
        [-0.8, 2.8],
        [2.6, -1.6],
    ])
    y_test = torch.tensor([0,1])
    # Creating instances of ToyDatasets
    train_ds = ToyDataset(X_train, y_train)
    test_ds = ToyDataset(X_test, y_test)
    # Creating the needed dataloaders
    train_dataloader = DataLoader(dataset=train_ds, batch_size=2, shuffle=False, pin_memory=True, sampler=DistributedSampler(train_ds))
    test_dataloader = DataLoader(dataset=test_ds, batch_size=2, shuffle=False)
    return train_dataloader, test_dataloader

# Wrapper for main
def main(rank, world_size, num_epochs):
    ddp_setup(rank, world_size)
    train_loader, test_loader = prepare_dataset()
    model = NeuralNetwork(num_inputs=2, num_outputs=2)
    model.to(rank)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.5)
    model = DDP(model, device_ids=[rank])

    for epoch in num_epochs:
        train_loader.sampler.set_epoch(epoch) # ensures each epoch gets samples ordered differently
        model.train()
        for features, labels in train_loader:
            features, labels = features.to(rank), labels.to(rank)
            logits = model(features) # forward pass
            loss = F.cross_entropy(logits, labels)
            optimizer.zero_grad() # prevets gradients accumulation
            loss.backward() # gradients are calculated from the computation graph. Note that DDP synchronizes and averages the gradients across all ranks
            optimizer.step() # the synched graients are used to update the model parameters as to minimize the loss
            # LOGGING
            print(f"[GPU{rank}] Epoch: {epoch+1:03d} | Batchsize {len(labels):03d} | Train/Val Loss: {loss:.2f}")
    model.eval()
    # compute accuracy
    train_acc = compute_accuracy(model, train_loader, device=rank)
    print(f"[GPU{rank} Training accuracy {train_acc}]")
    test_acc = compute_accuracy(model, test_loader, device=rank)
    print(f"[GPU{rank} Test accuracy {test_acc}]")
    destroy_process_group()

# computing accuracy
def compute_accuracy(model, dataloader, device):
    model = model.eval()
    correct = 0.0
    total_examples = 0

    for idx, (features, labels) in enumerate(dataloader):
        features, labels = features.to(device), labels.to(device)
        with torch.no_grad():
            logits = model(features)
        predictions = torch.argmax(logits, dim=1)
        compare = predictions==labels
        correct += torch.sum(compare)
        total_examples += len(compare)
    return (correct/total_examples).item() # .item() extracts the number from the tensor

if __name__ == "__main__":
    print("PyTorch version", torch.__version__)
    print("CUDA available:", torch.cuda.is_available())
    print("Number of GPUs available:", torch.cuda.device_count())
    torch.manual_seed(123)

    num_epochs = 3
    world_size = torch.cuda.device_count() # total number of GPUs
    mp.spawn(main, args=(world_size, num_epochs), nprocs=world_size) # spawn --> automatically passes the rank, nprocs=world_size --> means there will be one process per GPU
    