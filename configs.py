#!/usr/bin/env python3
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from hydra.core.config_store import ConfigStore
from hydra_plugins.hydra_submitit_launcher.config import SlurmQueueConf
from iqrl import iQRLConfig
from omegaconf import MISSING


@dataclass
class TrainConfig:
    """Training config used in train.py"""

    defaults: List[Any] = field(
        default_factory=lambda: [
            "_self_",
            {"agent": "iqrl"},
            {"env": "dog-run"},
            # Use submitit to launch slurm jobs on cluster w/ multirun
            {"override hydra/launcher": "slurm"},
            # Make the logging colourful
            {"override hydra/job_logging": "colorlog"},
            {"override hydra/hydra_logging": "colorlog"},
        ]
    )

    # Configure environment (overridden by defaults list)
    env_name: str = MISSING
    task_name: str = MISSING
    # env_name: str = "walker"
    # task_name: str = "walk"

    # Agent (overridden by defaults list)
    agent: iQRLConfig = field(default_factory=iQRLConfig)

    # Experiment
    max_episode_steps: int = 1000  # Max episode length
    num_episodes: int = 500  # Number of training episodes
    random_episodes: int = 10  # Number of random episodes at start
    action_repeat: int = 2
    buffer_size: int = 10_000_000
    prefetch: int = 5
    seed: int = 42
    checkpoint: Optional[str] = None  # /file/path/to/checkpoint
    device: str = "cuda"  # "cpu" or "cuda" etc
    verbose: bool = False  # if true print training progress

    # Evaluation
    eval_every_episodes: int = 20
    num_eval_episodes: int = 10
    capture_eval_video: bool = False  # Fails on AMD GPU so set to False
    capture_train_video: bool = False
    log_dormant_neuron_ratio: bool = False

    # W&B config
    use_wandb: bool = False
    wandb_project_name: str = "iqrl"
    run_name: str = "iqrl-${now:%Y-%m-%d_%H-%M-%S}"

    # Override the Hydra config to get better dir structure with W&B
    hydra: Any = field(
        default_factory=lambda: {
            "run": {"dir": "output/hydra/${hydra.job.name}/${now:%Y-%m-%d_%H-%M-%S}"},
            "verbose": False,
            "job": {"chdir": True},
            "sweep": {"dir": "${hydra.run.dir}", "subdir": "${hydra.job.num}"},
        }
    )


@dataclass
class SlurmConfig(SlurmQueueConf):
    """
    See here for config options
    https://github.com/facebookresearch/hydra/blob/main/plugins/hydra_submitit_launcher/hydra_plugins/hydra_submitit_launcher/config.py
    """

    timeout_min: int = 1440  # 24 hours
    mem_gb: int = 32
    name: str = "${env_name}-${task_name}"
    gres: str = "gpu:1"
    # signal_delay_s: int = 6000
    # constraint: "volta"
    stderr_to_stdout: bool = True
    # additional_parameters: List[Any] = field(
    #     default_factory=lambda: [
    #         {"mail-user": "scannell.aidan@gmail.com"},
    #         {"mail-type": "BEGIN"},  # send email when job begins
    #         {"mail-type": "END"},  # send email when job ends
    #         {"mail-type": "FAIL"},  # send email if job fails
    #     ]
    # )


@dataclass
class LUMIConfig(SlurmConfig):
    """
    See here for config options
    https://github.com/facebookresearch/hydra/blob/main/plugins/hydra_submitit_launcher/hydra_plugins/hydra_submitit_launcher/config.py
    """

    account: str = "project_462000623"
    partition: str = "small-g"  # Partition (queue) name
    timeout_min: int = 1440  # 24 hours


@dataclass
class EnvConfig:
    # Configure environment
    env_name: str = "dog"
    task_name: str = "run"


@dataclass
class EasyDMCConfig(EnvConfig):
    num_episodes: int = 2000

    # age
    agent: Any = field(
        default_factory=lambda: {
            "exploration_noise_num_steps": 50  # number of episodes do decay noise
        }
    )


@dataclass
class MediumDMCConfig(EnvConfig):
    num_episodes: int = 2000

    agent: Any = field(
        default_factory=lambda: {
            "exploration_noise_num_steps": 150  # number of episodes do decay noise
        }
    )


@dataclass
class HardDMCConfig(EnvConfig):
    num_episodes: int = 15000

    # agent: Any = field(
    #     default_factory=lambda: {
    #         "exploration_noise_num_steps": 500  # number of episodes do decay noise
    #     }
    # )


# class DogRunConfig(HardDMCConfig):
@dataclass
# class DogRunConfig:
class DogRunConfig(TrainConfig):
    # defaults: List[Any] = field(default_factory=lambda: ["train"])

    env_name: str = "dog"
    task_name: str = "run"

    # agent: Any = field(default_factory=lambda: {"latent_dim": 1024, "nstep": 3})
    # agent: iQRLConfig = field(default_factory=iQRLConfig(latent_dim=1024, nstep=3))
    # defaults: List[Any] = field(
    #     default_factory=lambda: [
    #         {"/agent": "iqrl"},
    #     ]
    # )
    # defaults: List[Any] = field(
    #     default_factory=lambda: [
    #         "train"
    #         # # {"env": "dog-run"},
    #         # # Use submitit to launch slurm jobs on cluster w/ multirun
    #         # {"override hydra/launcher": "slurm"},
    #         # # Make the logging colourful
    #         # {"override hydra/job_logging": "colorlog"},
    #         # {"override hydra/hydra_logging": "colorlog"},
    #         # "_self_",
    #     ]
    # )


cs = ConfigStore.instance()
cs.store(name="train", node=TrainConfig)
cs.store(name="iqrl", group="agent", node=iQRLConfig)
cs.store(name="slurm", group="hydra/launcher", node=SlurmConfig)
cs.store(name="lumi", group="hydra/launcher", node=LUMIConfig)

# cs.store(name="dog-run", group="env", node=DogRunConfig, package="_global_")
# cs.store(name="dog-run", group="env", node=DogRunConfig, package="_global_")
