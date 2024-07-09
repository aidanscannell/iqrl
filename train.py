#!/usr/bin/env python3
import os
from dataclasses import dataclass, field
from functools import partial
from typing import Optional

import hydra
import omegaconf
from agents.iqrl import iQRLConfig
from hydra.core.config_store import ConfigStore


@dataclass
class TrainConfig:
    env_name: str = "dog"
    task_name: str = "run"

    agent: iQRLConfig = field(default_factory=iQRLConfig)

    # Observation stuff
    from_pixels: bool = False
    pixels_only: bool = False
    num_frames_to_stack: int = 3  # only used for pixel observations

    # Experiment
    max_episode_steps: int = 500  # Max episode length (1000 steps as action_repeat=2)
    num_episodes: int = 500  # Number of training episodes
    random_episodes: int = 10  # Number of random episodes at start
    action_repeat: int = 2
    buffer_size: int = 10_000_000
    prefetch: int = 5
    seed: int = 42
    checkpoint: Optional[str] = None  # /file/path/to/checkpoint
    device: str = "cuda"  # "cpu" or "cuda" etc

    # Evaluation
    eval_every_episodes: int = 10
    num_eval_episodes: int = 10
    capture_eval_video: bool = True  # Fails on AMD GPU so set to False
    capture_train_video: bool = False
    log_dormant_neuron_ratio: bool = False

    # W&B config
    use_wandb: bool = False
    wandb_project_name: str = "iQRL"
    run_name: str = f"TD3"


cs = ConfigStore.instance()
cs.store(name="base_train_config", node=TrainConfig)
cs.store(name="base_iqrl", group="agent", node=iQRLConfig)


@hydra.main(version_base="1.3", config_path="./cfgs", config_name="train")
def train(cfg: TrainConfig):
    import logging
    import math
    import pprint
    import random
    import time

    import torch

    # This is needed to render videos on GPU
    if torch.cuda.is_available() and (cfg.device == "cuda"):
        os.environ["MUJOCO_GL"] = "osmesa"
        os.environ["PYOPENGL_PLATFORM"] = "osmesa"

    import agents
    import helper as h
    import numpy as np
    from envs import make_env
    from tensordict.nn import TensorDictModule
    from torchrl.data.tensor_specs import BoundedTensorSpec
    from torchrl.record.loggers.csv import CSVLogger
    from torchrl.record.loggers.wandb import WandbLogger
    from utils import ReplayBuffer

    logging.basicConfig(level=logging.DEBUG)
    logger = logging.getLogger(__name__)

    ###### Fix seed for reproducibility ######
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False

    cfg.device = (
        "cuda" if torch.cuda.is_available() and (cfg.device == "cuda") else "cpu"
    )
    cfg.agent.device = cfg.device
    logger.info(f"Using device: {cfg.device}")

    cfg_dict = omegaconf.OmegaConf.to_container(
        cfg, resolve=True, throw_on_missing=True
    )
    pprint.pprint(cfg_dict)

    ###### Initialise W&B ######
    writer = WandbLogger(
        exp_name=cfg.run_name,
        offline=not cfg.use_wandb,
        project=cfg.wandb_project_name,
        # log_dir="./logs",
        group=f"{cfg.env_name}-{cfg.task_name}",
        tags=[f"{cfg.env_name}-{cfg.task_name}", f"seed={str(cfg.seed)}"],
        # config=cfg_dict,
        #     monitor_gym=cfg.monitor_gym,
        save_code=True,
        # dir=os.path.join(get_original_cwd(), "output"),
        # dir="./logs",
    )
    writer.log_hparams(cfg)

    ###### Setup vectorized environment for training/evaluation/video recording ######
    make_env_fn = partial(
        make_env,
        env_name=cfg.env_name,
        task_name=cfg.task_name,
        seed=cfg.seed,
        frame_skip=cfg.action_repeat,
        num_frames_to_stack=cfg.num_frames_to_stack,
        from_pixels=cfg.from_pixels,
        pixels_only=cfg.pixels_only,
        device=cfg.device,
        # max_episode_steps=cfg.max_episode_steps,
    )
    env = make_env_fn(record_video=False)
    eval_env = make_env_fn(record_video=False)
    video_env = make_env_fn(record_video=cfg.capture_eval_video)

    assert isinstance(
        env.action_spec, BoundedTensorSpec
    ), "only continuous action space is supported"

    ###### Prepare replay buffer ######
    nstep = max(cfg.agent.get("nstep", 1), cfg.agent.get("horizon", 1))
    rb = ReplayBuffer(
        buffer_size=cfg.buffer_size,
        batch_size=cfg.agent.batch_size,
        nstep=nstep,
        gamma=cfg.agent.gamma,
        prefetch=cfg.prefetch,
        pin_memory=True,  # will be set to False if device=="cpu"
        device=cfg.device,
    )

    ###### Init agent ######
    agent = agents.iQRL(
        cfg=cfg.agent,
        obs_spec=env.observation_spec["observation"],
        act_spec=env.action_spec,
    )
    # Load state dict into this agent from filepath (or dictionary)
    if cfg.checkpoint is not None:
        state_dict = torch.load(cfg.checkpoint)
        agent.load_state_dict(state_dict["model"])
        logger.info(f"Loaded checkpoint from {cfg.checkpoint}")

    policy_module = TensorDictModule(
        lambda obs: agent.select_action(obs, eval_mode=False),
        in_keys=["observation"],
        out_keys=["action"],
    )
    eval_policy_module = TensorDictModule(
        lambda obs: agent.select_action(obs, eval_mode=True),
        in_keys=["observation"],
        out_keys=["action"],
    )

    env_step = 0
    start_time = time.time()
    for episode_idx in range(cfg.num_episodes):
        episode_start_time = time.time()
        ##### Rollout the policy in the environment #####
        with torch.no_grad():
            data = env.rollout(max_steps=cfg.max_episode_steps, policy=policy_module)
        if episode_idx == 0:
            print(f"data {data}")

        ##### Add data to the replay buffer #####
        rb.extend(data)

        ##### Log episode metrics #####
        num_new_transitions = data["next"]["step_count"][-1].cpu().item()
        env_step += num_new_transitions
        episode_reward = data["next"]["episode_reward"][-1].cpu().item()
        logger.info(
            f"Train | Return {episode_reward:.2f} | Env Step {env_step} | Episode {episode_idx}"
        )
        rollout_metrics = {
            "episodic_return": episode_reward,
            "episodic_return": episode_reward,
            "episodic_length": num_new_transitions,
            "env_step": env_step,
        }
        success = data["next"].get("success", None)
        if success is not None:
            episode_success = success.any()
            rollout_metrics.update({"episodic_success": episode_success})

        writer.log_scalar(name="rollout/", value=rollout_metrics)

        ##### Train agent (after collecting some random episodes) #####
        if episode_idx > cfg.random_episodes - 1:
            logger.info(
                f"Training agent w. {num_new_transitions} new data @ step {env_step}..."
            )
            train_metrics = agent.update(
                replay_buffer=rb, num_new_transitions=num_new_transitions
            )
            logger.info("Finished training agent.")

            ##### Log training metrics #####
            writer.log_scalar(name="train/", value=train_metrics)

            ##### Save checkpoint #####
            torch.save({"model": agent.state_dict()}, "./checkpoint")

            ###### Evaluate ######
            if episode_idx % cfg.eval_every_episodes == 0:
                ##### Calculate avg. episodic return (optionally avg. success) #####
                eval_metrics = {}
                with torch.no_grad():
                    episodic_returns, episodic_successes = [], []
                    for idx in range(cfg.num_eval_episodes):
                        print(f"Eval episode {idx}")
                        eval_data = eval_env.rollout(
                            max_steps=cfg.max_episode_steps, policy=eval_policy_module
                        )
                        episodic_returns.append(
                            eval_data["next"]["episode_reward"][-1].cpu().item()
                        )
                        success = eval_data["next"].get("success", None)
                        if success is not None:
                            episodic_successes.append(success.any())

                    episodic_return = sum(episodic_returns) / cfg.num_eval_episodes

                    if success is not None:
                        # TODO is episodic_successes being calculated correctly
                        episodic_success = (
                            sum(episodic_successes) / cfg.num_eval_episodes
                        )
                        eval_metrics.update({"episodic_success": episodic_success})
                    if cfg.capture_eval_video:
                        video_env.rollout(
                            max_steps=cfg.max_episode_steps, policy=eval_policy_module
                        )
                        video_env.transform.dump()

                ##### Eval metrics #####
                eval_metrics.update(
                    {
                        "episodic_return": episodic_return,
                        "elapsed_time": time.time() - start_time,
                        "SPS": int(env_step / (time.time() - start_time)),
                        "episode_time": time.time() - episode_start_time,
                        "env_step": env_step,
                        "episode": episode_idx,
                    }
                )
                logger.info(
                    f"Eval | Return {episode_reward:.2f} | Env step {env_step} | Episode {episode_idx} | SPS {eval_metrics['SPS']}"
                )

                ##### Log rank of latent and active codebook percent #####
                batch = rb.sample()
                eval_metrics.update(agent.metrics(batch))

                ##### Log metrics to W&B or csv #####
                writer.log_scalar(name="eval/", value=eval_metrics)

        # Release some GPU memory (if possible)
        torch.cuda.empty_cache()

    env.close()


if __name__ == "__main__":
    train()  # pyright: ignore