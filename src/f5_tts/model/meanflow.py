import logging
from typing import Callable, Optional

import torch
from torchdiffeq import odeint
import torch.nn as nn
log = logging.getLogger()

import torch
import torch.nn.functional as F
from einops import rearrange
from functools import partial
import numpy as np
import math


def normalize_to_neg1_1(x):
    return x * 2 - 1


def unnormalize_to_0_1(x):
    return (x + 1) * 0.5


def stopgrad(x):
    return x.detach()


def adaptive_l2_loss(error, gamma=0, c=1e-3):
    """
    Adaptive L2 loss: sg(w) * ||Δ||_2^2, where w = 1 / (||Δ||^2 + c)^p, p = 1 - γ
    Args:
        error: Tensor of shape (B, C, W, H)
        gamma: Power used in original ||Δ||^{2γ} loss
        c: Small constant for stability
    Returns:
        Scalar loss
    """
    delta_sq = torch.mean(error ** 2, dim=(1, 2), keepdim=False)
    p = 1.0 - gamma
    w = 1.0 / (delta_sq + c).pow(p)
    loss = delta_sq  # ||Δ||^2
    return stopgrad(w) * loss


def cosine_annealing(start, end, step, total_steps):
    cos_inner = math.pi * step / total_steps
    return end + 0.5 * (start - end) * (1 + math.cos(cos_inner))


class MeanFlow:
    def __init__(
        self, 
        steps=1,  
        flow_ratio=0.75,
        time_dist=['uniform', -0.4, 1.0],
        w=2.0,
        k=0.9,
        cfg_uncond='u',
        jvp_api='autograd',
    ):
        super().__init__()
        self.flow_ratio = flow_ratio
        self.time_dist = time_dist
        self.w = w
        self.k = k
        self.steps = steps
        
        self.cfg_uncond = cfg_uncond
        self.jvp_api = jvp_api
        assert jvp_api in ['funtorch', 'autograd'], "jvp_api must be 'funtorch' or 'autograd'"
        if jvp_api == 'funtorch':
            self.jvp_fn = torch.func.jvp
            self.create_graph = False
        elif jvp_api == 'autograd':
            self.jvp_fn = torch.autograd.functional.jvp
            self.create_graph = True
        log.info(f'MeanFlow initialized with {steps} steps')

    def sample_t_r(self, batch_size, device):
        if self.time_dist[0] == 'uniform':
            samples = np.random.rand(batch_size, 2).astype(np.float32)

        elif self.time_dist[0] == 'lognorm':
            mu, sigma = self.time_dist[-2], self.time_dist[-1]
            normal_samples = np.random.randn(batch_size, 2).astype(np.float32) * sigma + mu
            samples = 1 / (1 + np.exp(-normal_samples)) 

        t_np = np.maximum(samples[:, 0], samples[:, 1])
        r_np = np.minimum(samples[:, 0], samples[:, 1])

        num_selected = int(self.flow_ratio * batch_size)  
        indices = np.random.permutation(batch_size)[:num_selected]
        r_np[indices] = t_np[indices]

        t = torch.tensor(t_np, device=device)
        r = torch.tensor(r_np, device=device)
        return t, r
    
    def to_prior(self, fn: Callable, x1: torch.Tensor) -> torch.Tensor:
        return self.run_t0_to_t1(fn, x1)

    @torch.no_grad()
    def to_data(self, fn: Callable, x0: torch.Tensor) -> torch.Tensor:
        return self.run_t0_to_t1(fn, x0)
        
    def run_t0_to_t1(self, fn: Callable, x0: torch.Tensor) -> torch.Tensor:
        t = torch.ones((x0.shape[0],), device=x0.device,dtype=x0.dtype)
        r = torch.zeros((x0.shape[0],), device=x0.device,dtype=x0.dtype)
        steps = torch.linspace(1, 0, self.steps + 1).to(device=x0.device,dtype=x0.dtype)
        for ti, t in enumerate(steps[:-1]):
            t = t.expand(x0.shape[0])
            next_t = steps[ti + 1].expand(x0.shape[0])
            u_flow = fn(t=t, r=next_t, x=x0)
            dt = (t - next_t).mean()
            x0 = x0 - dt * u_flow
        return x0

    def loss(self,
            fn: Callable,
            x1: torch.Tensor,
            cond: torch.Tensor,
            text: torch.Tensor,
            mask: torch.Tensor,
            rand_span_mask: torch.Tensor,
            ):

        batch_size = x1.shape[0]
        device = x1.device
        x0 = torch.randn_like(x1)
        t, r = self.sample_t_r(batch_size, device)
        t_ = rearrange(t, "b -> b 1 1 ")
        r_ = rearrange(r, "b -> b 1 1 ")
        x_t = (1 - t_) * x0 + t_ * x1  # r < t
        flow = x1 - x0

    
        if self.w is not None:
            uncond =  torch.zeros_like(cond)
            uncond_text = torch.zeros_like(text)
            u_t = fn(x=x_t,
                     cond=uncond,
                     text=uncond_text, 
                     r=t,
                     t=t,
                     mask=mask).detach().requires_grad_(False)
        
            v_hat = self.w * flow + (1 - self.w) * u_t
        else:
            v_hat = flow

        model_partial = partial(fn, cond=cond, text=text, mask=mask)
        jvp_args = (
            lambda x_t, r, t: model_partial(x=x_t, r=r, t=t),
            (x_t, r, t),
            (v_hat, torch.zeros_like(r), torch.ones_like(t)),
        )
        if self.create_graph:
            u, dudt = self.jvp_fn(*jvp_args, create_graph=True)
        else:
            u, dudt = self.jvp_fn(*jvp_args)

        u_tgt = v_hat - (t_ - r_) * dudt
        error = u - stopgrad(u_tgt)
        loss = adaptive_l2_loss(error)
        loss = loss * rand_span_mask.unsqueeze(-1)
        return loss, u