#!/usr/bin/env python3
import abc
from typing import Optional

import torch.nn as nn
from jaxtyping import Float
from tensordict import TensorDict
from torch import Tensor
from torchrl.data import BoundedTensorSpec, CompositeSpec
from utils import ReplayBuffer


Observation = Float[Tensor, "obs_dim"]
# State = Float[Tensor, "state_dim"]
Latent = Float[Tensor, "latent_dim"]
Action = Float[Tensor, "action_dim"]
#
BatchObservation = Float[Observation, "batch"]
# BatchState = Float[State, "batch"]
BatchLatent = Float[Latent, "batch"]
BatchAction = Float[Action, "batch"]

Value = Float[Tensor, ""]
BatchValue = Float[Value, "batch_size"]


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
        # self.act_spec_low = act_spec.low.to(device)
        # self.act_spec_high = act_spec.high.to(device)

    @abc.abstractmethod
    def select_action(
        self, obsrvation: TensorDict, eval_mode: EvalMode = False
    ) -> Action:
        raise NotImplementedError

    @abc.abstractmethod
    def update(
        self, replay_buffer: ReplayBuffer, num_new_transitions: int
    ) -> Optional[dict]:
        raise NotImplementedError

    def metrics(self, batch) -> dict:
        return {}
