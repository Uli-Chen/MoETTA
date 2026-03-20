# MoETTA

This repository extends the original MoETTA framework by adding support for the CIFAR-10 and CIFAR-10-C datasets.

- CIFAR-10: https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz  
- CIFAR-10-C: https://zenodo.org/records/2535967/files/CIFAR-10-C.tar?download=1

## Set Up Environment

1. Run following command to set up codebase and Python environment.

```bash
git clone https://github.com/AnikiFan/MoETTA.git
cd MoETTA
# In case you haven't install uv, run following command if you are using Linux
# curl -LsSf https://astral.sh/uv/install.sh | sh
uv sync
uv run ray start --head
```

2. Create `.env` file under `MoETTA` directory.

```bash
RAY_ADDRESS=<YOUR RAY SERVER ADDRESS> # Get it by running `uv run ray start --head`
WANDB_API_KEY=<YOUR WANDB API KEY>
WANDB_BASE_URL=<YOUR WANDB SERVER URL> # If you are not using wandb-local, then fill it with `https://api.wandb.ai`
```

3. Tailor the environment configuration to yours.

The base configuration is located at `config/config.py`, where the configuration related to path needed to be changed according to your environment.
If you use CIFAR 10, also set `env.cifar10_data_path` and `env.cifar10c_data_path`.
By default, `env.cifar10_data_path` is `data`, so torchvision-style CIFAR-10 layout (`data/cifar-10-batches-py`) works out of the box.
If `data` contains `cifar-10-python.tar.gz`, it will be extracted automatically on first run.

1. Prepare CIFAR-10-C (if you want to run CIFAR-10-C evaluation).

Supported formats:

- `data/CIFAR-10-C/labels.npy` and `data/CIFAR-10-C/<corruption>.npy`
- `data/CIFAR-10-C/CIFAR-10-C/labels.npy` and `data/CIFAR-10-C/CIFAR-10-C/<corruption>.npy`
- `data/CIFAR-10-C.tar` (or `.tar.gz`) without manual extraction; it will be auto-extracted on first run.

The loader searches archives under both `env.cifar10c_data_path` and its parent directory, so the default setup works when you place the tar under `data/`.

## Run Experiment

```base
# Run an experiment locally, i.e., without ray
uv run main.py base --env.local

# Run an experiment with wandb offline
uv run main.py base --env.wandb_mode offline

# Run a hyper-parameter tuning/sweep by designating search space configuration
uv run main.py base --tune.search_space /home/fx25/workspace/MoETTA/config/search_space/seed.yaml

uv run main.py base --algo.algorithm eata

uv run main.py base --algo.algorithm moetta

uv run main.py base --algo.algorithm moetta --data.corruption potpourri+

# CIFAR-10
uv run main.py cifar10 --env.local

# CIFAR-10-C (default = mix over COMMON_CORRUPTIONS_15, severity = 5)
uv run main.py cifar10c --env.local

# CIFAR-10-C with one specific corruption
uv run main.py cifar10c --env.local --data.cifar_corruption gaussian_noise --data.level 5

```

## Add Configuration

Base configuration is located at `config/config.py`.

Derived configuration can be stored in `config/subconfigs/` and `config/subconfigs/potpourri.py` serves as an example.

To add a configuration, only two things need to be done:

1. Add a configuration file into `config/subconfigs/`
2. Import the added file into `config/__init__.py`
