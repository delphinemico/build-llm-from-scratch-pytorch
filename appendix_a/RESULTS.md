# Appendix A — Results

## Environment

| Item | Value |
|---|---|
| Date | |
| Python version | |
| PyTorch version | |
| Operating system | Windows |
| CPU | |
| GPU | |
| CUDA availability | |
| Number of GPUs | |

## Standard CPU training

### Command

```bash
python appendix_a/code/01_train_cpu.py
```

### Results

To be recorded.

## Single-GPU training

### Command

```bash
python appendix_a/code/02_train_single_gpu.py
```

### Results

To be recorded.

## Distributed Data Parallel training

### Command

```bash
torchrun --standalone --nproc_per_node=2 appendix_a/code/03_train_ddp.py
```

### Results

To be recorded.

## Errors encountered and resolutions

| Error or issue | Cause | Resolution |
|---|---|---|
| | | |

## Reproducibility notes

Record relevant details such as:

- random seed,
- batch size,
- number of epochs,
- learning rate,
- Python version,
- PyTorch version,
- CPU or GPU used,
- CUDA version,
- number of GPUs,
- operating-system-specific behavior.