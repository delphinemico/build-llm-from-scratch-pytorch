# Data

Datasets are stored locally and are not committed to this repository.

## Directory structure

- `raw/`: original downloaded or generated data
- `processed/`: transformed data used by experiments

The `raw/` and `processed/` directories are excluded from Git tracking.

Create them locally when needed:

```bash
mkdir -p data/raw data/processed
```

## Dataset documentation

For each dataset, document:

- the data source,
- how to obtain or generate it,
- expected filenames,
- preprocessing steps,
- licensing or usage restrictions,
- the chapters or experiments that use it.
