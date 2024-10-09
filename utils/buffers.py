# Code adapted from https://github.com/DLR-RM/stable-baselines3/blob/master/stable_baselines3/common/buffers.py
import logging
from typing import NamedTuple, Optional

import torch
from tensordict import TensorDict
from torchrl.data import ReplayBuffer as TorchRLReplayBuffer
from torchrl.data.replay_buffers import LazyMemmapStorage
from torchrl.data.replay_buffers.samplers import SliceSampler


logger = logging.getLogger(__name__)


class ReplayBufferSamples(NamedTuple):
    # observations: torch.Tensor
    observations: TensorDict
    actions: torch.Tensor
    # next_observations: torch.Tensor
    next_observations: TensorDict
    dones: torch.Tensor
    # timeouts: torch.Tensor
    terminateds: torch.Tensor
    # truncateds: torch.Tensor
    rewards: torch.Tensor
    next_state_gammas: torch.Tensor
    z: Optional[TensorDict]
    next_z: Optional[TensorDict]


class ReplayBuffer:
    def __init__(
        self,
        buffer_size: int,
        batch_size: int,
        nstep: int = 1,
        gamma: float = 0.99,
        prefetch: int = 10,
        pin_memory: bool = True,
        device: str = "cpu",
    ):
        if device == "cpu":
            pin_memory = False
            logger.info(f"On CPU so setting pin_memory=False")

        self.nstep = nstep
        self.gamma = gamma
        self.sampler = SliceSampler(
            slice_len=nstep,
            end_key=None,
            traj_key=("collector", "traj_ids"),
            truncated_key=None,
        )
        self.rb = TorchRLReplayBuffer(
            storage=LazyMemmapStorage(buffer_size, device=device),
            pin_memory=pin_memory,
            sampler=self.sampler,
            prefetch=prefetch,
            batch_size=batch_size * nstep,
            # transform=MultiStepTransform(n_steps=3, gamma=0.95),
        )
        self.batch_size = batch_size

    def extend(self, data):
        self.rb.extend(data.cpu())

    def sample(
        self, return_nstep: bool = False, batch_size: Optional[int] = None
    ) -> ReplayBufferSamples:
        batch = self._sample()
        if batch_size is not None and batch_size > self.batch_size:
            # TODO Fix this hack to get larger batch size
            # If requesting large batch size sample multiple times and concat
            for _ in range(batch_size // self.batch_size):
                batch = torch.cat([batch, self._sample()], 1)
            batch = batch[:, :batch_size]
        batch = ReplayBufferSamples(
            observations=batch["observation"],
            actions=batch["action"],
            next_observations=batch["next"]["observation"],
            dones=batch["next"]["done"][..., 0],
            terminateds=batch["next"]["terminated"][..., 0].to(torch.int),
            rewards=batch["next"]["reward"][..., 0],
            next_state_gammas=batch["next_state_gammas"],
            z=None,
            next_z=None,
        )
        if not return_nstep:
            return batch
        else:
            return to_nstep(batch, nstep=self.nstep, gamma=self.gamma)

    def _sample(self) -> TensorDict:
        batch = self.rb.sample().view(-1, self.nstep).transpose(0, 1)
        next_state_gammas = torch.ones_like(
            batch["next"]["done"][..., 0], dtype=torch.float32
        )
        batch.update({"next_state_gammas": next_state_gammas}, inplace=True)
        return batch


@torch.no_grad()
def to_nstep(
    batch: ReplayBufferSamples, nstep: int, gamma: float = 0.99
) -> ReplayBufferSamples:
    """Form n-step samples (truncate if timeout)"""
    if nstep > 1:
        dones = torch.zeros_like(batch.dones[0], dtype=torch.bool)
        terminateds = torch.zeros_like(batch.terminateds[0], dtype=torch.bool)
        rewards = torch.zeros_like(batch.rewards[0])
        next_state_gammas = torch.ones_like(batch.dones[0], dtype=torch.float32)
        next_obs = torch.zeros_like(batch.observations[0])
        next_z = torch.zeros_like(batch.next_z[0]) if batch.next_z is not None else None
        for t in range(nstep):
            next_obs = torch.where(
                dones[..., None], next_obs, batch.next_observations[t]
            )
            if next_z is not None:
                next_z = torch.where(dones[..., None], next_z, batch.next_z[t])
            dones = torch.logical_or(dones, batch.dones[t])
            next_state_gammas *= torch.where(dones, 1, gamma)
            terminateds *= torch.where(
                dones, terminateds, torch.logical_or(terminateds, batch.terminateds[t])
            )
            rewards += torch.where(dones, 0, gamma**t * batch.rewards[t])
        nstep_batch = ReplayBufferSamples(
            observations=batch.observations[0],
            actions=batch.actions[0],
            next_observations=next_obs,
            dones=dones.to(torch.int),
            terminateds=terminateds.to(torch.int),
            rewards=rewards,
            next_state_gammas=next_state_gammas,
            z=batch.z[0] if batch.z is not None else None,
            next_z=next_z,
        )
    else:
        # TODO Can remove this else
        nstep_batch = ReplayBufferSamples(
            observations=batch.observations[0],
            actions=batch.actions[0],
            next_observations=batch.next_observations[0],
            dones=batch.dones[0],
            terminateds=batch.terminateds[0],
            rewards=batch.rewards[0],
            next_state_gammas=batch.next_state_gammas[0],
            z=batch.z[0] if batch.z is not None else None,
            next_z=batch.next_z[0] if batch.next_z is not None else None,
        )
    return nstep_batch
