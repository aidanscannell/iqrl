#!/usr/bin/env python3
from typing import Optional

import gymnasium as gym
from dm_control import suite
from metaworld.envs import ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE
from torchrl.envs import DMControlEnv, GymEnv, StepCounter, TransformedEnv
from torchrl.envs.transforms import (
    CatFrames,
    CatTensors,
    Compose,
    DoubleToFloat,
    RenameTransform,
    Resize,
    RewardSum,
    ToTensorImage,
    TransformedEnv,
)
from torchrl.record import VideoRecorder
from torchrl.record.loggers.csv import CSVLogger

from .dmcontrol import make_env as dmcontrol_make_env
from .metaworld import make_env as metaworld_make_env


def make_env(
    env_name: str,
    task_name: Optional[str] = None,
    seed: int = 42,
    from_pixels: bool = True,
    frame_skip: int = 2,
    pixels_only: bool = False,
    render_size: int = 64,
    num_frames_to_stack: int = 1,
    logger=None,
    record_video: bool = False,
    device: str = "cpu",
):
    if not from_pixels:
        pixels_only = False

    if env_name in gym.envs.registry.keys():
        env = GymEnv(
            env_name=env_name,
            from_pixels=from_pixels,
            frame_skip=frame_skip,
            pixels_only=pixels_only,
            device=device,
        )
    elif (env_name, task_name) in suite.ALL_TASKS or env_name == "cup":
        env = dmcontrol_make_env(
            env_name=env_name,
            task_name=task_name,
            from_pixels=from_pixels or record_video,
            frame_skip=frame_skip,
            pixels_only=pixels_only,
            device=device,
        )
    elif (
        env_name.split("-", 1)[-1] + "-v2-goal-observable"
        in ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE
    ):
        env = metaworld_make_env(
            env_name=env_name.split("-", 1)[-1] + "-v2-goal-observable",
            from_pixels=from_pixels or record_video,
            seed=seed,
            frame_skip=frame_skip,
            pixels_only=pixels_only,
            device=device,
        )

    if not pixels_only:
        env = TransformedEnv(
            env,
            Compose(
                RenameTransform(in_keys=["observation"], out_keys=["state"]),
                RenameTransform(in_keys=["state"], out_keys=[("observation", "state")]),
            ),
        )
    env = TransformedEnv(
        env,
        Compose(DoubleToFloat(), StepCounter(), RewardSum()),
    )

    if from_pixels:
        env = TransformedEnv(
            env,
            Compose(
                ToTensorImage(in_keys="pixels"),
                Resize(render_size, render_size),
                # RenameTransform(in_keys="pixels", out_keys=("observation", "pixels")),
                RenameTransform(
                    in_keys=["pixels"], out_keys=[("observation", "pixels")]
                ),
                CatFrames(
                    N=num_frames_to_stack, dim=-3, in_keys=("observation", "pixels")
                ),
            ),
        )
        video_rec_in_keys = ("observation", "pixels")
    else:
        video_rec_in_keys = "pixels"
    if record_video:
        if logger is None:
            logger = CSVLogger(exp_name="", log_dir="./logs", video_format="mp4")
        env = TransformedEnv(
            env,
            VideoRecorder(logger=logger, tag="run_video", in_keys=video_rec_in_keys),
        )
    env.set_seed(seed)
    return env
