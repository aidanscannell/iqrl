#!/usr/bin/env python3
import copy
import logging
import math
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import helper as h
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import utils
import wandb
from custom_types import Agent, EvalMode
from helper import soft_update_params
from tensordict import TensorDict
from torchrl.data import BoundedTensorSpec, CompositeSpec
from utils import ReplayBuffer, ReplayBufferSamples


logger = logging.getLogger(__name__)


@dataclass
class iQRLConfig:
    """Config for iQRL"""

    """MLP dims for actor/critic/dynamics"""
    mlp_dims: List[int] = field(default_factory=lambda: [512, 512])
    """Learning rate for actor/critic"""
    lr: float = 3e-4
    """Batch size - same for for representation learning and actor/critic"""
    batch_size: int = 256
    """Number of parameter updates per new data, i.e .UTD ratio """
    utd_ratio: int = 1
    """Update actor less frequently than critic"""
    actor_update_freq: int = 2
    """Discount factor"""
    gamma: float = 0.99
    """Target network update rate"""
    tau: float = 0.005
    """Number of critics"""
    num_critics: int = 2
    """Use N-step returns for Q-learning?"""
    nstep: int = 1  # nstep returns
    """What observation types to use? ["state"] or ["pixels"] or ["state", "pixels"]"""
    obs_types: List[str] = field(default_factory=lambda: ["state"])
    """ENCODER CONFIG"""
    """Size of latent space"""
    latent_dim: int = 512
    """Horizon used for representation learning"""
    horizon: int = 5
    """Discount factor for representation learning"""
    rho: float = 0.9
    """MLP dims for encoder/decoder"""
    enc_mlp_dims: List[int] = field(default_factory=lambda: [256])
    """Learning rate for encoder/dynamics/projection/reward"""
    enc_lr: float = 1e-4
    """Momentum coefficient for target encoder"""
    enc_tau: float = 0.005
    """Update encoder less frequently than actor/critic"""
    enc_update_freq: int = 1
    """Clips the gradient norm of the encoder"""
    grad_clip_norm: Optional[int] = 20
    """Use target encoder for representation learning"""
    use_tar_enc: bool = True
    """Predict change in latent or next latent? i.e. next_z = z + f(z, a) else next_z = f(z, a)"""
    use_delta: bool = True
    """Use LayerNorm or BatchNorm for encoder?"""
    enc_norm_type: str = "ln"
    """Use temporal consistency loss for representation learning"""
    use_tc_loss: bool = True
    """Use reward prediction for representation learning"""
    use_rew_loss: bool = False
    """Use cosine similarity for consistency loss - otherwise MSE"""
    use_cosine_similarity_dynamics: bool = True
    """Flag to turn FSQ on/off """
    use_fsq: bool = True
    """FSQ levels - setting as [8,8] corresponds to a codebook of size 8*8=62=2^8"""
    fsq_levels: List[int] = field(default_factory=lambda: [8, 8])
    """PROJECTION HEAD"""
    """Project the latent using an MLP before calculating the temporal consistency loss?"""
    use_latent_projection: bool = False
    """MLP dims for projection head"""
    projection_mlp_dims: List[int] = field(default_factory=lambda: [256])
    """Dimension of projection - defaults to latent_dim/16"""
    proj_dim: Optional[int] = None
    """EXPLORATION NOISE SCHEDULE"""
    """Initial variance"""
    exploration_noise_start: float = 1.0
    """Final variance"""
    exploration_noise_end: float = 0.1
    """Number of episodes do decay noise"""
    exploration_noise_num_steps: int = 50
    """POLICY SMOOTHING"""
    """Variance"""
    policy_noise: float = 0.2
    """Clip the noise"""
    noise_clip: float = 0.3
    """OTHER"""
    """Logging frequency"""
    logging_freq: int = 100
    """If True try to compile all NNs"""
    compile: bool = False
    """All NNs will be put on this device"""
    device: str = "cuda"
    """Print training losses?"""
    verbose: bool = False
    name: str = "iQRL"


class Actor(nn.Module):
    def __init__(
        self,
        cfg: iQRLConfig,
        act_dim: int,
        action_scale,
        action_bias,
        act_low,
        act_high,
    ):
        super().__init__()
        self.cfg = cfg
        self.action_scale = action_scale
        self.action_bias = action_bias
        self.act_low = act_low
        self.act_high = act_high
        self.mlp = h.mlp(self.cfg.latent_dim, self.cfg.mlp_dims, act_dim)

    def forward(self, z):
        a = self.mlp(z)
        a = torch.tanh(a)
        a = a * self.action_scale + self.action_bias
        return a


class Critic(nn.Module):
    def __init__(self, cfg: iQRLConfig, act_dim: int):
        super().__init__()
        self.cfg = cfg
        qs = [
            h.mlp(cfg.latent_dim + act_dim, mlp_dims=cfg.mlp_dims, out_dim=1).to(
                cfg.device
            )
            for _ in range(cfg.num_critics)
        ]
        for q in qs:
            h.orthogonal_init(q.parameters())

        self.qs = h.Ensemble(qs)

    def forward(self, z, a, return_type: str = "all"):
        x = torch.cat([z, a], -1)
        qs = self.qs(x)
        if return_type == "all":
            return qs
        if return_type == "min":
            return torch.min(qs, 0)[0]
        elif return_type == "avg":
            return torch.mean(qs, 0)
        else:
            raise NotImplementedError(
                f"return_type should be 'all' or 'min' or 'avg' not {return_type}"
            )


class Encoder(nn.Module):
    def __init__(
        self, cfg: iQRLConfig, obs_spec: CompositeSpec, act_spec: BoundedTensorSpec
    ):
        super().__init__()
        self.cfg = cfg
        self.obs_spec = obs_spec
        self.act_spec = act_spec
        obs_dim = np.array(obs_spec["state"].shape).prod().item()
        act_dim = np.array(act_spec.shape).prod().item()

        ##### Configure FSQ stuff #####
        if cfg.use_fsq:
            self.num_channels = len(cfg.fsq_levels)
            if not cfg.latent_dim % self.num_channels == 0:
                raise NotImplementedError(
                    "latent_dim must be divisible by number of FSQ channels"
                )
            self._fsq = h.FSQ(levels=cfg.fsq_levels)
            self.cfg.latent_dim *= self.num_channels

        ##### Init encoder #####
        self._encoder = nn.ModuleDict()
        if "state" in cfg.obs_types:  # Encoder for state-based observations
            self._encoder.update(
                {
                    "state": h.mlp(
                        obs_dim,
                        cfg.enc_mlp_dims,
                        cfg.latent_dim,
                    )
                }
            )
        if "pixels" in cfg.obs_types:  # Encoder for pixel-based observations
            if self.cfg.enc_norm_type == "bn":
                raise NotImplementedError("Need to implement BN for CNN encoder")
            self._encoder.update(
                {
                    "pixels": h.CNNEncoder(
                        obs_shape=obs_spec.shape,
                        latent_dim=cfg.latent_dim,
                        hidden_dim=256,
                        frame_diff=False,
                    )
                }
            )
        if cfg.use_tar_enc:
            self._encoder_tar = copy.deepcopy(self._encoder).requires_grad_(False)

        self._trans = h.mlp(cfg.latent_dim + act_dim, cfg.mlp_dims, cfg.latent_dim)

        if cfg.use_latent_projection:
            if cfg.proj_dim is None:
                cfg.proj_dim = int(self.cfg.latent_dim / 16)
            self._proj = h.mlp(cfg.latent_dim, cfg.mlp_dims, cfg.proj_dim)
            if cfg.use_tar_enc:
                self._proj_tar = copy.deepcopy(self._proj).requires_grad_(False)

        if cfg.use_rew_loss:
            self._reward = h.mlp(cfg.latent_dim + act_dim, cfg.mlp_dims, 1)

    def encode(self, obs, tar: bool = False):
        if "pixels" in self.cfg.obs_types:
            raise NotImplementedError()
        zs = {}
        for key in obs.keys():
            if tar:
                zs.update({key: self._encoder_tar[key](obs[key])})
            else:
                zs.update({key: self._encoder[key](obs[key])})
        if "state" in self.cfg.obs_types and "pixels" not in self.cfg.obs_types:
            z = zs["state"]
            td = TensorDict({"state": z}, batch_size=obs.batch_size)
        elif "state" not in self.cfg.obs_types and "pixels" in self.cfg.obs_types:
            z = zs["pixels"]
        else:
            raise NotImplementedError("Need to make encoder take both state and pixels")

        td = TensorDict({"state": z}, batch_size=obs.batch_size)
        if self.cfg.use_fsq:
            td.update(self.quantize(z))
        return td

    def trans(self, z, a):
        za = torch.concat([z, a], -1)
        delta_z = self._trans(za)
        next_z = z + delta_z if self.cfg.use_delta else delta_z
        return next_z

    def reward(self, z, a):
        za = torch.concat([z, a], -1)
        r = self._reward(za)
        return r

    def project(self, z, tar: bool = False):
        """Project latent state before calculating consistency loss"""
        z = self._proj_tar(z) if tar else self._proj(z)
        return z

    def quantize(self, z):
        """Quantize the latent state"""
        return self._fsq(z)

    def loss(self, batch: ReplayBufferSamples) -> Tuple[torch.Tensor, dict]:
        tc_loss = torch.zeros(1).to(self.cfg.device)
        reward_loss = torch.zeros(1).to(self.cfg.device)

        a = batch.actions

        ##### Create targets #####
        with torch.no_grad():
            next_obs = batch.next_observations
            zs_tar = self.encode(next_obs, tar=True)
            zs_tar = zs_tar["state"]

        ##### Latent rollout #####
        zs = torch.empty_like(zs_tar)
        z = self.encode(batch.observations[0])["state"]
        dones = torch.zeros_like(batch.dones[0], dtype=torch.bool)
        terminateds_or_dones = torch.zeros_like(batch.dones, dtype=torch.bool)
        for t in range(self.cfg.horizon):
            dones = torch.where(terminateds_or_dones[t], dones, batch.dones[t])
            terminateds_or_dones[t] = torch.logical_or(
                terminateds_or_dones[t], torch.logical_or(dones, batch.terminateds[t])
            )

            # Predict next latent
            next_z_pred = self.trans(z=z, a=a[t])
            if self.cfg.use_fsq:
                next_z_pred = self.quantize(next_z_pred)["state"]

            # Don't forget this
            z = next_z_pred

            zs[t] = z

        rho = torch.tensor([self.cfg.rho**t for t in range(self.cfg.horizon)]).to(
            self.cfg.device
        )
        terminateds_or_dones = terminateds_or_dones.to(torch.int)

        ##### (Optional) Reward prediction loss #####
        if self.cfg.use_rew_loss:
            r_tar = batch.rewards[..., None]  # Reward target
            r_pred = self.reward(z=zs, a=a)
            assert r_pred.ndim == 3 and r_tar.ndim == 3
            _reward_loss = (r_pred[..., 0] - r_tar[..., 0]) ** 2
            _rho_reward_loss = rho * torch.mean(
                (1 - terminateds_or_dones) * _reward_loss, -1
            )
            reward_loss = torch.mean(_rho_reward_loss)

        ##### (Optional) Project latent before consistency loss #####
        if self.cfg.use_latent_projection:
            zs_tar = self.projection_tar(zs_tar)
            zs = self.projection(zs)

        ##### Temporal consistency loss #####
        if self.cfg.use_tc_loss:
            if self.cfg.use_cosine_similarity_dynamics:
                """Cosine similarity"""
                _tc_loss = nn.CosineSimilarity(dim=-1, eps=1e-6)(zs, zs_tar)
            else:
                """Mean squared error"""
                _tc_loss = torch.mean((zs - zs_tar) ** 2, dim=-1)
            _rho_tc_loss = rho * torch.mean((1 - terminateds_or_dones) * _tc_loss, -1)
            tc_loss = torch.mean(_rho_tc_loss)

        loss = tc_loss + reward_loss
        info = {
            "tc_loss": tc_loss.item(),
            "reward_loss": reward_loss.item(),
            "enc_loss": loss.item(),
            "z_min": torch.min(zs).item(),
            "z_max": torch.max(zs).item(),
            "z_mean": torch.mean(zs.to(torch.float)).item(),
            "z_median": torch.median(zs).item(),
        }
        return loss, info

    def metrics(self, batch):
        z = self.encode(batch.observations[0])

        # Calculate rank of latent
        metrics = h.calc_rank(name="z", z=z["state"])

        # Calculate percentage of codebook that's active
        if self.cfg.use_fsq:
            num_codes = torch.tensor(math.prod(self.cfg.fsq_levels), device=z.device)

            def act_percent_fn(z):
                # TODO can't vmap this because Tensor.unique() not allowed in vmap
                return z.unique().numel() / num_codes * 100

            active_percents = torch.empty(z["indices"].shape[1])
            for i in range(z["indices"].shape[1]):
                active_percents[i] = act_percent_fn(z["indices"][i])
            metrics.update(
                {
                    # "active_percent": active_percent,
                    "active_percent_avg": active_percents.mean(),
                    "active_percent_min": active_percents.min(),
                    "active_percent_max": active_percents.max(),
                }
            )

        # TODO add dormant neuron ratio stuff
        # metrics.update(h.calc_dormant_neuron_ratio(batch, agent=self))

        return metrics

    def train(self):
        self._encoder.train()
        self._trans.train()
        if self.cfg.use_rew_loss:
            self._reward.train()
        if self.cfg.use_latent_projection:
            self._proj.train()

    def eval(self):
        self._encoder.eval()
        self._trans.eval()
        if self.cfg.use_rew_loss:
            self._reward.eval()
        if self.cfg.use_latent_projection:
            self._proj.eval()


class iQRL(Agent):
    def __init__(
        self, cfg: iQRLConfig, obs_spec: CompositeSpec, act_spec: BoundedTensorSpec
    ):
        super().__init__(
            obs_spec=obs_spec, act_spec=act_spec, device=cfg.device, name=cfg.name
        )

        if "pixels" in cfg.obs_types:
            raise NotImplementedError

        ##### Calculate dimensions for MLPs #####
        act_dim = np.array(act_spec.shape).prod().item()
        if "state" in cfg.obs_types:
            obs_dim = np.array(obs_spec["state"].shape).prod().item()
        else:
            raise NotImplementedError("Need to use state observations")

        self.cfg = cfg

        ##### Init encoder #####
        self.encoder = Encoder(cfg, obs_spec=obs_spec, act_spec=act_spec).to(cfg.device)
        if cfg.compile:
            self.encoder = torch.compile(self.encoder, mode="default")
        self.enc_opt = torch.optim.AdamW(self.encoder.parameters(), lr=cfg.enc_lr)

        ##### Init actor network and its target network #####
        self._pi = Actor(
            cfg,
            act_dim=act_dim,
            action_scale=(act_spec.high - act_spec.low).to(cfg.device) / 2.0,
            action_bias=(act_spec.high + act_spec.low).to(cfg.device) / 2.0,
            act_low=act_spec.low,
            act_high=act_spec.high,
        ).to(cfg.device)
        self._pi = torch.compile(self._pi, mode="default") if cfg.compile else self._pi
        pi_tar = copy.deepcopy(self._pi).requires_grad_(False)
        self._pi_tar = torch.compile(pi_tar, mode="default") if cfg.compile else pi_tar

        ##### Init critics and their target networks #####
        Q = Critic(cfg, act_dim=act_dim).to(cfg.device)
        self.Q = torch.compile(Q, mode="default") if cfg.compile else Q
        Q_tar = copy.deepcopy(self.Q).requires_grad_(False)
        self.Q_tar = torch.compile(Q_tar, mode="default") if cfg.compile else Q_tar

        ##### Optimizers #####
        self.pi_opt = torch.optim.Adam(self._pi.parameters(), lr=cfg.lr)
        self.q_opt = torch.optim.Adam(self.Q.parameters(), lr=cfg.lr)

        ##### Exploration noise schedule #####
        self._exploration_noise_schedule = h.LinearSchedule(
            start=cfg.exploration_noise_start,
            end=cfg.exploration_noise_end,
            num_steps=cfg.exploration_noise_num_steps,
        )

        # Counters for number of param updates
        self.critic_update_counter = 0
        self.pi_update_counter = 0

    def update(self, replay_buffer: ReplayBuffer, num_new_transitions: int) -> dict:
        """Update representation and TD3 at same time"""
        num_updates = int(num_new_transitions * self.cfg.utd_ratio)
        info = {}

        if self.cfg.verbose:
            logger.info(f"Performing {num_updates} iQRL updates...")
        for i in range(num_updates):
            batch = replay_buffer.sample()

            # Update enc less frequently than actor/critic
            if i % self.cfg.enc_update_freq == 0:
                info.update(self.representation_update_step(batch=batch))

            # Map observations to latent
            with torch.no_grad():
                z = self.encoder.encode(batch.observations, tar=False)
                next_z = self.encoder.encode(batch.next_observations, tar=False)
            batch = batch._replace(z=z, next_z=next_z)

            ##### Make nstep returns #####
            if self.cfg.horizon == 1:
                raise NotImplementedError("Check N-step batch is made correctly if h=1")
            nstep_batch = utils.to_nstep(
                batch, nstep=self.cfg.nstep, gamma=self.cfg.gamma
            )

            ##### Update critic #####
            info.update(self.critic_update_step(batch=nstep_batch))

            ##### Update actor less frequently than critic #####
            if self.critic_update_counter % self.cfg.actor_update_freq == 0:
                info.update(self.pi_update_step(batch=nstep_batch))

            if i % self.cfg.logging_freq == 0:
                if self.cfg.verbose:
                    logger.info(
                        f"Iteration {i} | loss {info['enc_loss']:.3} | tc loss {info['tc_loss']:.3} | reward loss {info['reward_loss']:.3}"
                    )
                if wandb.run is not None:
                    wandb.log(info)

        ###### Log some stuff ######
        if wandb.run is not None:
            wandb.log({"exploration_noise": self.exploration_noise})

        self._exploration_noise_schedule.step()

        if self.cfg.verbose:
            logger.info("Finished training iQRL")
        return info

    # @torch.compile
    def representation_update_step(self, batch: ReplayBufferSamples):
        self.encoder.train()
        loss, info = self.encoder.loss(batch=batch)

        self.enc_opt.zero_grad(set_to_none=True)
        loss.backward()

        if self.cfg.grad_clip_norm is not None:
            enc_params = list(self.encoder.parameters())
            grad_norm = torch.nn.utils.clip_grad_norm_(
                enc_params, self.cfg.grad_clip_norm, error_if_nonfinite=False
            )
            info.update({"grad_norm": float(grad_norm)})

        self.enc_opt.step()

        # Update the tar network
        soft_update_params(
            self.encoder._encoder, self.encoder._encoder_tar, tau=self.cfg.enc_tau
        )
        if self.cfg.use_latent_projection:
            soft_update_params(
                self.encoder._proj, self.encoder._proj_tar, tau=self.cfg.enc_tau
            )

        self.encoder.eval()
        return info

    def critic_update_step(self, batch: ReplayBufferSamples):
        self.critic_update_counter += 1
        self.Q.train()

        # Check batch shapes
        assert batch.rewards.ndim == 1
        assert batch.rewards.shape[0] == batch.observations.shape[0]
        assert batch.z is not None
        assert batch.next_z is not None

        # Make Q target
        with torch.no_grad():
            z = batch.z["state"]
            next_z = batch.next_z["state"]
            a = batch.actions
            a_next = self.pi(next_z, tar=True, eval_mode=True, smooth=True)

            min_q_next_tar = self.Q_tar(z=next_z, a=a_next, return_type="min")[..., 0]

            assert min_q_next_tar.shape == batch.rewards.shape
            next_q_value = (
                batch.rewards
                + (1 - batch.terminateds) * batch.next_state_gammas * min_q_next_tar
            )

        q1_values, q2_values = self.Q(z=z, a=a, return_type="all")

        q1_loss = F.mse_loss(q1_values[..., 0], next_q_value)
        q2_loss = F.mse_loss(q2_values[..., 0], next_q_value)
        q_loss = q1_loss + q2_loss

        ##### Optimize critic #####
        self.q_opt.zero_grad(set_to_none=True)
        q_loss.backward()
        self.q_opt.step()

        ##### Update the target network #####
        soft_update_params(self.Q, self.Q_tar, tau=self.cfg.tau)

        self.Q.eval()
        return {
            "q_loss": q_loss.item() / 2,
            "q1_loss": q1_loss.item(),
            "q2_loss": q2_loss.item(),
            "q1_values": q1_values.mean().item(),
            "q2_values": q2_values.mean().item(),
            "critic_update_counter": self.critic_update_counter,
        }

    def pi_update_step(self, batch: ReplayBufferSamples):
        self.pi_update_counter += 1
        self._pi.train()

        z = batch.z["state"]
        pi_loss = -self.Q(z=z, a=self._pi(z), return_type="min").mean()

        ##### Optimize actor #####
        self.pi_opt.zero_grad(set_to_none=True)
        pi_loss.backward()
        self.pi_opt.step()

        ##### Update the target network #####
        soft_update_params(self._pi, self._pi_tar, tau=self.cfg.tau)

        self._pi.eval()
        return {
            "actor_loss": pi_loss.item(),
            "actor_update_counter": self.pi_update_counter,
        }

    @torch.no_grad()
    def select_action(self, obs, eval_mode: EvalMode = False):
        is_flat_obs = False
        if obs.batch_size == torch.Size([]):
            obs = obs.view(1)
            is_flat_obs = True

        z = self.encoder.encode(obs, tar=False).to(torch.float)

        a = self.pi(z["state"], tar=False, eval_mode=eval_mode)

        a = a[0] if is_flat_obs else a
        return a

    def pi(self, z, tar: bool = False, eval_mode: bool = False, smooth: bool = False):
        a = self._pi_tar(z) if tar else self._pi(z)
        if not eval_mode:
            a += torch.normal(0, self._pi.action_scale * self.exploration_noise)
        if smooth:
            clipped_noise = (
                torch.randn_like(a, device=self.cfg.device) * self.cfg.policy_noise
            ).clamp(-self.cfg.noise_clip, self.cfg.noise_clip) * self._pi.action_scale
            a += clipped_noise
        a = a.clamp(self.act_spec_low, self.act_spec_high)
        return a

    @property
    def exploration_noise(self):
        return self._exploration_noise_schedule()

    def metrics(self, batch):
        metrics = self.encoder.metrics(batch)

        metrics.update({"enc": h.calc_mean_opt_moments(self.enc_opt)})
        metrics.update({"Q": h.calc_mean_opt_moments(self.q_opt)})
        metrics.update({"pi": h.calc_mean_opt_moments(self.pi_opt)})

        # TODO add dormant neuron ratio stuff
        # metrics.update(h.calc_dormant_neuron_ratio(batch, agent=self))

        return metrics

    @property
    def total_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
