#!/usr/bin/env python3
from dataclasses import dataclass, field
from typing import Any, List, Optional

from hydra.core.config_store import ConfigStore
from iqrl import iQRLConfig


@dataclass
class TrainConfig:
    env_name: str = "dog"
    task_name: str = "run"

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
    defaults: List[Any] = field(
        default_factory=lambda: [
            # Use submitit to launch slurm jobs on cluster w/ multirun
            {"override hydra/launcher": "slurm"},
            # Make the logging colourful
            {"override hydra/job_logging": "colorlog"},
            {"override hydra/hydra_logging": "colorlog"},
        ]
    )


@dataclass
class SlurmConfig:
    # class SlurmConfig(SlurmQueueConf):
    _target_: str = (
        "hydra_plugins.hydra_submitit_launcher.submitit_launcher.SlurmLauncher"
    )
    # defaults: List[Any] = field(
    #     default_factory=lambda: [
    #         "submitit_slurm"
    #         # Use submitit to launch slurm jobs on cluster w/ multirun
    #         # {"override hydra/launcher": "slurm"},
    #         # Make the logging colourful
    #         # {"override hydra/job_logging": "colorlog"},
    #         # {"override hydra/hydra_logging": "colorlog"},
    #     ]
    # )
    # defaults:
    # - submitit_slurm

    # _target_: hydra_plugins.hydra_submitit_launcher.submitit_launcher.SlurmLauncher
    account: str = "project_462000462"
    partition: str = "small-g"  # Partition (queue) name
    timeout_min: int = 2880  # 48 hours
    tasks_per_node: int = 1
    mem_gb: int = 64
    nodes: int = 1
    name: str = "${env_name}-${task_name}"
    gres: str = "gpu:1"
    signal_delay_s: int = 6000
    max_num_timeout: int = 0
    # additional_parameters: dict = {}
    array_parallelism: int = 256
    # constraint: "volta"
    stderr_to_stdout: bool = True


cs = ConfigStore.instance()
cs.store(name="train", node=TrainConfig)
cs.store(name="iqrl", group="agent", node=iQRLConfig)
cs.store(name="slurm", group="hydra/launcher", node=SlurmConfig)
