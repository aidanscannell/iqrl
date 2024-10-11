# iQRL: implicitly Quantized Representations for Reinforcement Learning
This repository is the official implementation of [iQRL](www.aidanscannell.com/iqrl), a reinforcement learning algorithm for continous control.

Learning representations for reinforcement learning (RL) has shown much promise for continuous control. We propose an efficient representation learning method using only a self-supervised latent-state consistency loss. Our approach employs an encoder and a dynamics model to map observations to latent states and predict future latent states, respectively. We achieve high performance and prevent representation collapse by quantizing the latent representation such that the rank of the representation is empirically preserved. Our method, named iQRL: implicitly Quantized Reinforcement Learning, is straightforward, compatible with any model-free RL algorithm, and demonstrates excellent performance by outperforming other recently proposed representation learning methods in continuous control benchmarks from DeepMind Control Suite.

## Instructions

### Install
Install dependencies:
```sh
conda env create -f environment.yml
conda activate iqrl
```
You might need to install PyTorch with CUDA/ROCm.

### Running experiments
Train the agent:
``` sh
python train.py +env=walker-walk
```
To log metrics with W&B:
``` sh
python train.py +env=walker-walk ++use_wandb=True
```
All tested tasks are listed in`cfgs/env`.

### Configuring experiments
This repo uses hydra for configuration.
You can easily try new hyperparameters for `iQRL` by overriding them on the command line, for example,
``` sh
python train.py +env=walker-walk ++use_wandb=True ++agent.batch_size=512
```
changes the batch size to be 512 instead of default value found in `iqrl.py/iQRLConfig`.

You can also use hydra to submit multiple SLURM jobs directly from the command line using
``` sh
python train.py -m +env=walker-walk ++use_wandb=True ++agent.batch_size=256,512 ++agent.lr=1e-4,1e-4
```
This uses the `utils/cluster_utils.py/SlurmConfig` to configure the jobs, setting `timeout_min=1440` (i.e. 24hrs) and `mem_gb=32`.
If you want to run the job for longer (e.g 48hrs), you can use the following
``` sh
python train.py -m +env=walker-walk ++use_wandb=True ++agent.batch_size=256,512 ++agent.lr=1e-4,1e-4 ++hydra.launcher.timeout_min=2880
```

# BibTeX
Please consider citing our arXiv paper:
``` bibtex
@misc{scannell2024iqrl,
  title           = {iQRL - Implicitly Quantized Representations for Sample-efficient Reinforcement Learning},
  author          = {Aidan Scannell and Kalle Kujanpää and Yi Zhao and Mohammadreza Nakhaei and Arno Solin and Joni Pajarinen},
  year            = {2024},
  eprint          = {2406.02696},
  archivePrefix   = {arXiv},
  primaryClass    = {cs.LG}
}
```
and our workshop paper:
``` bibtex
@inproceedings{scannellQuantized2024,
  title           = {Quantized Representations Prevent Dimensional Collapse in Self-predictive {RL}},
  booktitle       = {ICML Workshop on Aligning Reinforcement Learning Experimentalists and Theorists ({ARLET})},
  author          = {Aidan Scannell and Kalle Kujanpää and Yi Zhao and Mohammadreza Nakhaei and Arno Solin and Joni Pajarinen},
  year            = {2024},
}
```
