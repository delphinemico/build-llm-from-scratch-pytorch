# Environment Setup

## Compute environments

Different environments are used for the three training configurations in this
repository.

| Training configuration | Execution environment |
|---|---|
| Standard CPU training | Local Windows computer |
| Single-GPU training | Google Colab GPU runtime |
| Multi-GPU DDP training | RunPod instance with two GPUs |

The local Windows environment is intentionally configured for CPU execution.
GPU-specific results will be recorded separately.

## Create the virtual environment

The virtual environment is stored outside the repository to keep Windows
installation paths short.

In Git Bash, run:

```bash
mkdir -p ~/venvs
python -m venv ~/venvs/llmbookvenv
```

## Activate it in Git Bash on Windows

```bash
source ~/venvs/llmbookvenv/Scripts/activate
```

After activation, the terminal prompt should begin with:

```text
(llmbookvenv)
```

## Virtual-environment location

The local virtual environment is stored at:

```text
C:\Users\delmi\venvs\llmbookvenv
```

The environment remains dedicated to this repository even though it is stored
outside the repository directory.

The virtual environment is not committed to Git. Exact installed package
versions are recorded in `requirements-lock.txt`.

## Verify the selected interpreter

```bash
python -c "import sys; print(sys.executable)"
```

The output should be:

```text
C:\Users\delmi\venvs\llmbookvenv\Scripts\python.exe
```

## Upgrade the packaging tools

```bash
python -m pip install --upgrade pip setuptools wheel
```

## Install PyTorch locally

The local environment uses CPU execution:

```bash
python -m pip install torch torchvision
```

The official PyTorch installation selector should be consulted when installing
PyTorch on a different machine or when CUDA support is required.

## Install the project and development dependencies

From the repository root, run:

```bash
python -m pip install -e ".[dev]"
```

The editable installation keeps the environment connected to the local source
code, so changes to the repository's Python modules are available without
reinstalling the project after every edit.

## Install notebook support

```bash
python -m pip install ipykernel
```

## Configure VS Code

Select this interpreter for the repository workspace:

```text
C:\Users\delmi\venvs\llmbookvenv\Scripts\python.exe
```

In VS Code:

1. Open the Command Palette with `Ctrl+Shift+P`.
2. Select `Python: Select Interpreter`.
3. Select `Enter interpreter path`.
4. Choose the `python.exe` shown above.

For Jupyter notebooks, select the same environment separately using the
notebook's kernel selector.

## Verify the installation

```bash
python --version
python -c "import torch; print('PyTorch version:', torch.__version__)"
python -c "import torch; print('CUDA available:', torch.cuda.is_available())"
python -c "import torch; print(torch.rand(2, 2))"
pytest
```

For the local CPU environment, the expected CUDA result is:

```text
CUDA available: False
```

This is expected and does not indicate an installation problem.

## Record the exact package versions

From the repository root:

```bash
python -m pip freeze > requirements-lock.txt
```

## Deactivate the virtual environment

```bash
deactivate
```

## Reopen the project later

At the beginning of a future study session:

```bash
cd ~/Documents/LEARNING/Manning_Learning/repos/build-llm-from-scratch-pytorch
source ~/venvs/llmbookvenv/Scripts/activate
```