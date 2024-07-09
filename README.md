# iQRL: implicitly Quantized Reinforcement Learning

## Instructions
Install dependencies:
```sh
conda env create -f environment.yml
conda activate iqrl
```
You might need to install PyTorch with CUDA/ROCm.
Train the agent:
``` sh
python train.py +env=walker-walk
```
To log metrics with W&B:
``` sh
python train.py +env=walker-walk ++use_wandb=True
```
All tested tasks are listed in`cfgs/env`
