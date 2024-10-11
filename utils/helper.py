#!/usr/bin/env python3
import copy
from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import utils
from functorch import combine_state_for_ensemble
from torch.func import functional_call, stack_module_state
from torch.linalg import cond, matrix_rank

# from utils.batch_renorm import BatchRenorm1d
from vector_quantize_pytorch import FSQ as _FSQ


def soft_update_params(model, model_target, tau: float):
    """Update slow-moving average of online network (target network) at rate tau."""
    with torch.no_grad():
        for params, params_target in zip(model.parameters(), model_target.parameters()):
            params_target.data.lerp_(params.data, tau)
            # One below is from CleanRL
            # params_target.data.copy_(tau * params.data + (1 - tau) * params_target.data)


def mlp(
    in_dim,
    mlp_dims,
    out_dim,
    act_fn=None,
    dropout=0.0,
    norm_mode: str = "ln",
    norm_after_act: bool = False,
):
    """
    MLP with LayerNorm, Mish activations, and optionally dropout.

    Adapted from https://github.com/tdmpc2/tdmpc2-eval/blob/main/helper.py
    """
    if isinstance(mlp_dims, int):
        mlp_dims = [mlp_dims]

    dims = [int(in_dim)] + mlp_dims + [int(out_dim)]
    mlp = nn.ModuleList()
    for i in range(len(dims) - 2):
        mlp.append(
            NormedLinear(
                dims[i],
                dims[i + 1],
                dropout=dropout * (i == 0),
                norm_mode=norm_mode,
                norm_after_act=norm_after_act,
            )
        )
    mlp.append(
        NormedLinear(
            dims[-2],
            dims[-1],
            act=act_fn,
            norm_mode=norm_mode,
            norm_after_act=norm_after_act,
        )
        if act_fn
        else nn.Linear(dims[-2], dims[-1])
    )
    return nn.Sequential(*mlp)


class FSQ(nn.Module):
    """
    Finite Scalar Quantization
    """

    def __init__(self, levels: List[int]):
        super().__init__()
        self.levels = levels
        self.num_channels = len(levels)
        self._fsq = _FSQ(levels)

    def forward(self, z):
        shp = z.shape
        z = z.view(*shp[:-1], -1, self.num_channels)
        if z.ndim > 3:  # TODO this might not work for CNN
            codes, indices = torch.func.vmap(self._fsq)(z)
        else:
            codes, indices = self._fsq(z)
        return {
            "codes": codes,
            "codes_flat": codes.flatten(-2),
            "indices": indices,
            "z": z,
            "state": codes.flatten(-2),
        }

    def __repr__(self):
        return f"FSQ(levels={self.levels})"


class NormedLinear(nn.Linear):
    """
    Linear layer with LayerNorm, Mish activation, and optionally dropout.

    Adapted from https://github.com/tdmpc2/tdmpc2-eval/blob/main/helper.py
    """

    def __init__(
        self,
        *args,
        dropout=0.0,
        act=nn.Mish(inplace=True),
        norm_mode: Optional[str] = "ln",  # "ln" or "bn" or "brn" or None
        norm_after_act: bool = True,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        if norm_mode == "bn":
            self.norm = nn.BatchNorm1d(self.out_features)
        elif norm_mode == "brn":
            self.norm = BatchRenorm1d(self.out_features)
        elif norm_mode == "ln":
            self.norm = nn.LayerNorm(self.out_features)
        elif norm_mode == None:
            self.norm = lambda x: x
        else:
            raise NotImplementedError(
                f"norm_mode should be 'ln', 'bn', 'brn' or None, not {norm_mode}"
            )

        self.norm_after_act = norm_after_act
        self.act = act
        self.dropout = nn.Dropout(dropout, inplace=True) if dropout else None

    def forward(self, x):
        x = super().forward(x)
        if self.dropout:
            x = self.dropout(x)
        if self.norm_after_act:
            return self.norm(self.act(x))
        else:
            return self.act(self.norm(x))

    def __repr__(self):
        repr_dropout = f", dropout={self.dropout.p}" if self.dropout else ""
        return f"NormedLinear(in_features={self.in_features}, \
        out_features={self.out_features}, \
        bias={self.bias is not None}{repr_dropout}, \
        act={self.act.__class__.__name__})"


class Ensemble(nn.Module):
    """
    Vectorized ensemble of modules.

    Adapted from https://github.com/nicklashansen/tdmpc2/blob/main/tdmpc2/common/layers.py
    """

    def __init__(self, modules, **kwargs):
        super().__init__()
        modules = nn.ModuleList(modules)
        fn, params, _ = combine_state_for_ensemble(modules)
        self.vmap = torch.vmap(
            fn, in_dims=(0, 0, None), randomness="different", **kwargs
        )
        self.params = nn.ParameterList([nn.Parameter(p) for p in params])
        self._repr = str(modules)

    def forward(self, *args, **kwargs):
        return self.vmap([p for p in self.params], (), *args, **kwargs)

    def __repr__(self):
        return "Vectorized " + self._repr


class EnsembleNew(nn.Module):
    """Vectorized ensemble of modules"""

    def __init__(self, modules, **kwargs):
        super().__init__()

        self.params_dict, self._buffers = stack_module_state(modules)
        self.params = nn.ParameterList([p for p in self.params_dict.values()])

        # Construct a "stateless" version of one of the models. It is "stateless" in
        # the sense that the parameters are meta Tensors and do not have storage.
        base_model = copy.deepcopy(modules[0])
        base_model = base_model.to("meta")

        def fmodel(params, buffers, x):
            return functional_call(base_model, (params, buffers), (x,))

        self.vmap = torch.vmap(
            fmodel, in_dims=(0, 0, None), randomness="different", **kwargs
        )
        self._repr = str(modules)

    def forward(self, *args, **kwargs):
        return self.vmap(self._get_params_dict(), self._buffers, *args, **kwargs)

    def _get_params_dict(self):
        params_dict = {}
        for key, value in zip(self.params_dict.keys(), self.params):
            params_dict.update({key: value})
        return params_dict

    def __repr__(self):
        return "Vectorized " + self._repr


@torch.no_grad()
def orthogonal_init(m):
    """Orthogonal layer initialization."""
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight.data)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    # elif isinstance(m, EnsembleLinear):
    #     for w in m.weight.data:
    #         nn.init.orthogonal_(w)
    #     if m.bias is not None:
    #         for b in m.bias.data:
    #             nn.init.zeros_(b)
    elif isinstance(m, (nn.Conv3d, nn.Conv2d, nn.ConvTranspose2d)):
        gain = nn.init.calculate_gain("relu")
        nn.init.orthogonal_(m.weight.data, gain)
        # nn.init.kaiming_uniform_(m.weight.data, mode='fan_in', nonlinearity='relu')
        if m.bias is not None:
            nn.init.zeros_(m.bias)


class LinearSchedule:
    def __init__(self, start: float, end: float, num_steps: int):
        self.start = start
        self.end = end
        self.num_steps = num_steps
        self.step_idx = 0
        self.values = np.linspace(start, end, num_steps)

    def __call__(self):
        return self.values[self.step_idx]

    def step(self):
        if self.step_idx < self.num_steps - 1:
            self.step_idx += 1


@torch.no_grad()
def calc_rank(name, z):
    """Log rank of latent"""
    rank3 = matrix_rank(z, atol=1e-3, rtol=1e-3)
    rank2 = matrix_rank(z, atol=1e-2, rtol=1e-2)
    rank1 = matrix_rank(z, atol=1e-1, rtol=1e-1)
    condition = cond(z)
    info = {}
    full_rank = z.shape[-1]
    for j, rank in enumerate([rank1, rank2, rank3]):
        rank_percent = rank.item() / full_rank * 100
        info.update({f"{name}-rank-{j}": rank.item()})
        info.update({f"{name}-rank-percent-{j}": rank_percent})
    info.update({f"{name}-cond-num": condition.item()})
    return info


def calc_dormant_neuron(inputs, model, opt, name, redo_tau=0.025):
    redo_out = utils.redo.run_redo(
        inputs=inputs,
        model=model,
        optimizer=opt,
        tau=redo_tau,  # 0.025 for default, else 0.1
        re_initialize=False,
        use_lecun_init=None,
    )
    return {
        f"{name}_dormant_t={redo_tau}_fraction": redo_out["dormant_fraction"],
        f"{name}_dormant_t={redo_tau}_count": redo_out["dormant_count"],
        f"{name}_dormant_t=0.0_fraction": redo_out["zero_fraction"],
        f"{name}_dormant_t=0.0_count": redo_out["zero_count"],
    }


def calc_dormant_neuron_ratio(batch, agent):
    import agents

    obs = batch.observations[0]
    action = batch.actions[0]
    if agent.use_state:
        key = "state"
    elif agent.use_pixels:
        key = "pixels"

    info = {}
    if isinstance(agent, agents.iQRL):
        obs = agent.encode(obs)
        if agent.use_act_enc:
            action = agent.encode_act(action)["action"]

        # Log dormant neuron ratio for encoder
        info.update(
            calc_dormant_neuron(
                inputs=batch.observations[0][key],
                model=agent.encoder[key],
                opt=agent.enc_opt,
                name="enc",
            )
        )

        if agent.use_tc_loss:
            # Log dormant neuron ratio for dynamics
            za = torch.concat([obs["state"], action], -1)
            info.update(
                calc_dormant_neuron(
                    inputs=za,
                    # inputs={"observation": z_batch, "action": action},
                    model=agent.dynamics,
                    opt=agent.enc_opt,
                    name="dynamics",
                )
            )

    # Log dormant neuron ratio for critic
    info.update(
        calc_dormant_neuron(
            inputs={"observation": obs["state"], "action": action},
            model=agent.critic,
            opt=agent.q_opt,
            name="q",
        )
    )

    # Log dormant neuron ratio for actor
    info.update(
        calc_dormant_neuron(
            inputs=obs["state"],
            model=agent.pi,
            opt=agent.pi_opt,
            name="actor",
        )
    )
    return info


def calc_mean_opt_moments(opt):
    first_moment, second_moment = 0, 0
    for group in opt.param_groups:
        for p in group["params"]:
            state = opt.state[p]
            try:
                first_moment += torch.sum(state["exp_avg"]) / len(state["exp_avg"])
                second_moment += torch.sum(state["exp_avg_sq"]) / len(state["exp_avg"])
            except KeyError:
                pass
    return {"first_moment_mean": first_moment, "second_moment_mean": second_moment}
