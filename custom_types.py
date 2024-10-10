#!/usr/bin/env python3
import abc
from typing import Optional

import torch.nn as nn
from tensordict import TensorDict
from torch import Tensor
from torchrl.data import BoundedTensorSpec, CompositeSpec
from utils import ReplayBuffer, ReplayBufferSamples


EvalMode = bool
T0 = bool


class Agent(abc.ABC, nn.Module):
    def __init__(
        self,
        obs_spec: CompositeSpec,
        act_spec: BoundedTensorSpec,
        device: str,
        name: str = "BaseAgent",
    ):
        super().__init__()

        self.obs_spec = obs_spec
        self.act_spec = act_spec
        self.device = device
        self.name = name
        self.register_buffer("act_spec_low", act_spec.low.to(device))
        self.register_buffer("act_spec_high", act_spec.high.to(device))

    @abc.abstractmethod
    def select_action(
        self, obsrvation: TensorDict, eval_mode: EvalMode = False
    ) -> Tensor:
        raise NotImplementedError

    @abc.abstractmethod
    def update(
        self, replay_buffer: ReplayBuffer, num_new_transitions: int
    ) -> Optional[dict]:
        raise NotImplementedError

    def metrics(self, batch: ReplayBufferSamples) -> dict:
        return {}
