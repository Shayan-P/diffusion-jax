import jax
import flax.linen as nn
import jax.numpy as jnp
import typing as tp
import logging

from unet import Unet, ConditionalUnet
from tqdm import tqdm
from functools import partial


class Scheduler:
    def __init__(self, num_timesteps, beta_start, beta_end):
        self.num_timesteps = num_timesteps
        self.beta_start = beta_start
        self.beta_end = beta_end

        self.betas = jnp.linspace(self.beta_start, self.beta_end, self.num_timesteps)
        self.alphas = 1 - self.betas
        self.alpha_bars = jnp.cumprod(self.alphas)
        self.sqrt_alpha_bars = jnp.sqrt(self.alpha_bars)
        self.sqrt_comp_alpha_bars = jnp.sqrt(1 - self.alpha_bars)

        if self.sqrt_alpha_bars[-1] > 1e-3:
            logging.warning("Final value of sqrt_alpha_bars is too high: %f", self.sqrt_alpha_bars[-1])


class DDPM(nn.Module):
    scheduler: Scheduler
    dims: tp.Tuple[int]
    timestep_num: int = 1000
    timestep_dim: int = 256
    channels: int = 1

    def setup(self):
        self.unet = Unet(timestep_num=self.timestep_num,
                    timestep_dim=self.timestep_dim,
                    out_channels=self.channels,
                    dims=self.dims)

    def __call__(self, x, t, noise):
        x_t = self.forward_diffusion(x, t, noise)
        noise_pred = self.unet(x_t, t, training=True)
        return noise_pred

    def forward_diffusion(self, x, t, noise):
        n, h, w, c = x.shape
        assert x.shape == noise.shape
        assert x.shape[0] == t.shape[0]
        assert len(t.shape) == 1

        alpha1 = jnp.take(self.scheduler.sqrt_alpha_bars, t)[:, None, None, None]
        alpha2 = jnp.take(self.scheduler.sqrt_comp_alpha_bars, t)[:, None, None, None]
        xt = alpha1 * x + alpha2 * noise
        return xt

    def reverse_diffusion(self,x_t,t,noise):
        n, h, w, c = x_t.shape
        assert x_t.shape[0] == t.shape[0]
        assert noise.shape == x_t.shape
        assert len(t.shape) == 1

        alpha1 = jnp.sqrt(1/self.scheduler.alphas.take(t))[:, None, None, None]
        alpha2 = ((1 - self.scheduler.alphas.take(t)) / self.scheduler.sqrt_comp_alpha_bars.take(t))[:, None, None, None]
        noise_pred = self.unet(x_t, t, training=False)
        mu = alpha1 * (x_t - alpha2 * noise_pred)

        alpha_bars_prev = self.scheduler.alpha_bars.take(t-1)
        alpha_bars = self.scheduler.alpha_bars.take(t)
        sigma_t = jnp.sqrt(
            self.scheduler.betas.take(t) * alpha_bars_prev / alpha_bars
        )[:, None, None, None]
        cond = jnp.where(t > 0, 1.0, 0.0)[:, None, None, None]
        return mu + sigma_t * noise * cond
    
    def sampling(self, rng, n_samples, data_shape):
        shape = (n_samples, *data_shape)

        def scan_fn(carry, t):
            rng, x_t = carry
            rng, step_rng = jax.random.split(rng)
            noise = jax.random.normal(step_rng, shape)
            t = jnp.repeat(t, n_samples)
            x_new = self.reverse_diffusion(x_t, t, noise)
            return (rng, x_new), None
        ts = jnp.arange(self.timestep_num)[::-1]
        key, rng = jax.random.split(rng)
        x_t = jax.random.normal(key, shape)
        (_, x_final), _ = jax.lax.scan(scan_fn, (rng, x_t), ts)
        return x_final

    def train_step(self, state, rng, x):
        n, h, w, c = x.shape
        t = jax.random.randint(rng, (n,), 0, self.timestep_num)
        noise = jax.random.normal(rng, x.shape)
        
        def loss_fn(params):
            noise_pred, updates = state.apply_fn(params, x, t, noise, mutable='batch_stats')
            return jnp.mean((noise - noise_pred) ** 2), updates
        
        value_grad = jax.value_and_grad(loss_fn, has_aux=True)
        (loss, updates), grad = value_grad(state.params)
        state.params.update(updates)
        state = state.apply_gradients(grads=grad)
        return state, loss


class DDPMConditional(nn.Module):
    scheduler: Scheduler
    dims: tp.Tuple[int]
    label_count: int
    label_dim: int = 64
    timestep_num: int = 1000
    timestep_dim: int = 256
    channels: int = 1

    def setup(self):
        self.unet = ConditionalUnet(timestep_num=self.timestep_num,
                    timestep_dim=self.timestep_dim,
                    out_channels=self.channels,
                    dims=self.dims,
                    label_count=self.label_count,
                    label_dim=self.label_dim)

    def __call__(self, x, labels, t, noise):
        x_t = self.forward_diffusion(x, t, noise)
        noise_pred = self.unet(x_t, labels, t, training=True)
        return noise_pred

    def forward_diffusion(self, x, t, noise):
        n, h, w, c = x.shape
        assert x.shape == noise.shape
        assert x.shape[0] == t.shape[0]
        assert len(t.shape) == 1

        alpha1 = jnp.take(self.scheduler.sqrt_alpha_bars, t)[:, None, None, None]
        alpha2 = jnp.take(self.scheduler.sqrt_comp_alpha_bars, t)[:, None, None, None]
        xt = alpha1 * x + alpha2 * noise
        return xt

    def reverse_diffusion(self, x_t, labels, t, noise):
        n, h, w, c = x_t.shape
        assert x_t.shape[0] == t.shape[0]
        assert noise.shape == x_t.shape
        assert len(t.shape) == 1

        alpha1 = jnp.sqrt(1/self.scheduler.alphas.take(t))[:, None, None, None]
        alpha2 = ((1 - self.scheduler.alphas.take(t)) / self.scheduler.sqrt_comp_alpha_bars.take(t))[:, None, None, None]
        noise_pred = self.unet(x_t, labels, t, training=False)
        mu = alpha1 * (x_t - alpha2 * noise_pred)

        alpha_bars_prev = self.scheduler.alpha_bars.take(t-1)
        alpha_bars = self.scheduler.alpha_bars.take(t)
        sigma_t = jnp.sqrt(
            self.scheduler.betas.take(t) * alpha_bars_prev / alpha_bars
        )[:, None, None, None]
        cond = jnp.where(t > 0, 1.0, 0.0)[:, None, None, None]
        return mu + sigma_t * noise * cond
    
    def sampling(self, rng, labels, data_shape):
        n_samples, = labels.shape
        shape = (n_samples, *data_shape)

        def scan_fn(carry, t):
            rng, x_t, labels = carry
            rng, step_rng = jax.random.split(rng)
            noise = jax.random.normal(step_rng, shape)
            t = jnp.repeat(t, n_samples)
            x_new = self.reverse_diffusion(x_t, labels, t, noise)
            return (rng, x_new, labels), None
        ts = jnp.arange(self.timestep_num)[::-1]
        key, rng = jax.random.split(rng)
        x_t = jax.random.normal(key, shape)
        (_, x_final, _), _ = jax.lax.scan(scan_fn, (rng, x_t, labels), ts)
        return x_final

    def train_step(self, state, rng, x, labels):
        n, h, w, c = x.shape
        t = jax.random.randint(rng, (n,), 0, self.timestep_num)
        noise = jax.random.normal(rng, x.shape)
        
        def loss_fn(params):
            noise_pred, updates = state.apply_fn(params, x, labels, t, noise, mutable='batch_stats')
            return jnp.mean((noise - noise_pred) ** 2), updates
        
        value_grad = jax.value_and_grad(loss_fn, has_aux=True)
        (loss, updates), grad = value_grad(state.params)
        state.params.update(updates)
        state = state.apply_gradients(grads=grad)
        return state, loss


class MLP(nn.Module):
    dims: tp.Tuple[int]

    # todo make sure this works...
    @nn.compact
    def __call__(self, x, y, t, training: bool):
        for dim in self.dims:
            x = jnp.concatenate([x, y, t], axis=-1) # late fusion?
            x = nn.Dense(dim)(x)
            x = nn.silu(x)
            x = nn.BatchNorm(use_running_average=not training)(x)
        x = nn.Dense(self.dims[-1])(x)
        return x


class DDPMConditionalWithMLP(nn.Module):
    scheduler: Scheduler
    dims: tp.Tuple[int]
    timestep_num: int = 1000
    timestep_dim: int = 256

    def setup(self):
        self.mlp = MLP(
            dims=self.dims
        )
        self.time_emb = nn.Embed(self.timestep_num, self.timestep_dim)
    
    def __call__(self, x, y, t, noise):
        x_t = self.forward_diffusion(x, t, noise)
        inp = jnp.concatenate([x_t, y], axis=-1)
        noise_pred = self.mlp(x_t, y, self.time_emb(t), training=True)
        return noise_pred

    def forward_diffusion(self, x, t, noise):
        n, c = x.shape
        assert x.shape == noise.shape
        assert x.shape[0] == t.shape[0]
        assert len(t.shape) == 1

        alpha1 = jnp.take(self.scheduler.sqrt_alpha_bars, t)[:, None]
        alpha2 = jnp.take(self.scheduler.sqrt_comp_alpha_bars, t)[:, None]
        xt = alpha1 * x + alpha2 * noise
        return xt

    def reverse_diffusion(self, x_t, y, t, noise):
        n, c = x_t.shape
        assert x_t.shape[0] == t.shape[0]
        assert noise.shape == x_t.shape
        assert len(t.shape) == 1

        alpha1 = jnp.sqrt(1/self.scheduler.alphas.take(t))[:, None]
        alpha2 = ((1 - self.scheduler.alphas.take(t)) / self.scheduler.sqrt_comp_alpha_bars.take(t))[:, None]
        noise_pred = self.mlp(x_t, y, self.time_emb(t), training=False)
        mu = alpha1 * (x_t - alpha2 * noise_pred)

        alpha_bars_prev = self.scheduler.alpha_bars.take(t-1)
        alpha_bars = self.scheduler.alpha_bars.take(t)
        sigma_t = jnp.sqrt(
            self.scheduler.betas.take(t) * alpha_bars_prev / alpha_bars
        )[:, None]
        cond = jnp.where(t > 0, 1.0, 0.0)[:, None]
        return mu + sigma_t * noise * cond
    
    def sampling(self, rng, y, data_shape):
        n_samples, _ = y.shape
        shape = (n_samples, *data_shape)

        def scan_fn(carry, t):
            rng, x_t, y = carry
            rng, step_rng = jax.random.split(rng)
            noise = jax.random.normal(step_rng, shape)
            t = jnp.repeat(t, n_samples)
            x_new = self.reverse_diffusion(x_t, y, t, noise)
            return (rng, x_new, y), None
        ts = jnp.arange(self.timestep_num)[::-1]
        key, rng = jax.random.split(rng)
        x_t = jax.random.normal(key, shape)
        (_, x_final, _), _ = jax.lax.scan(scan_fn, (rng, x_t, y), ts)
        return x_final

    def train_step(self, state, rng, x, y):
        n, c = x.shape
        t = jax.random.randint(rng, (n,), 0, self.timestep_num)
        noise = jax.random.normal(rng, x.shape)
        
        def loss_fn(params):
            noise_pred, updates = state.apply_fn(params, x, y, t, noise, mutable='batch_stats')
            return jnp.mean((noise - noise_pred) ** 2), updates
        
        value_grad = jax.value_and_grad(loss_fn, has_aux=True)
        (loss, updates), grad = value_grad(state.params)
        state.params.update(updates)
        state = state.apply_gradients(grads=grad)
        return state, loss
