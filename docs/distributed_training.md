# DDP Conceptual Explanation

* **Data Parallelism:** Model is replicated across GPUs. Each GPU gets the same replica of the model, but a different batch. Later, the gradients are synched/averaged.

* **Distributed Data Parallelism (DDP):** is a specific implementation of Data Parallelism in PyTorch. Synchronization of gradients happens during the backward pass, using **all-reduce** across all GPUs.

> A launcher (`torch.multiprocessing as mp`) must start multiple processes for DDP. In this script, `mp.spawn` is used.  
> DDP normally runs one process per GPU. Each process has its own Python interpreter, model replica, optimizer, and local training data.
> The spawn will automatically pass the rank which is a unique process ID.

→ During DDP setup, the process group is initialized  
→ Each process creates the model and moves its model replica to its assigned GPU.  
→ The model is wrapped in `DistributedDataParallel`. DDP registers the mechanisms needed to synchronize gradients when `loss.backward()` is called.  
→ A `DistributedSampler` gives each process a different shard of the training dataset.  
→ We loop over the epochs.  
→ At the beginning of each epoch, we call `train_loader.sampler.set_epoch(epoch)` so that shuffled samples can be ordered differently across epochs.  
→ Each process loops over its own local mini-batches.  
→ The forward pass produces logits.     
→ A loss is computed by comparing the logits with the target labels.   
→ We call optimizer.zero_grad() to prevent gradients accumulation.     
→ We call `loss.backward()` to calculate gradients from the computation graph. During this backward pass, DDP synchronizes and averages the gradients across all ranks.  
→ Each process calls `optimizer.step()`, which uses the synchronized gradients to update its local model replica. Because every process receives the same averaged gradients, the model replicas remain synchronized.  The synched gradients are used to update the model parameters in a way that minimizes the loss.  
→ After training, we call `model.eval()` and compute the training and test accuracy inside `torch.no_grad()`.   
→ Finally, the distributed process group is destroyed.  
