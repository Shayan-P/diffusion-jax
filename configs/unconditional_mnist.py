import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

from dataclasses import dataclass, field
from configs.base_conf import BaseConfig, BaseModelConfig
from model import MNISTDiffusion, Scheduler
from trainer import AbstractTrainer
from functools import partial
from pathlib import Path


@partial(jax.jit, static_argnames=("n_samples",))
def sampling(state, rng, n_samples):
    return state.apply_fn(state.params, method=MNISTDiffusion.sampling, n_samples=n_samples, rng=rng)


@partial(jax.jit, static_argnames=("model",))
def train_step(model, state, rng, x):
    return model.train_step(state=state, rng=rng, x=x)


class Trainer(AbstractTrainer):
  def __init__(self, model, log_path: Path):
    super().__init__()
    self.model = model
    self.log_path = log_path

  def train_batch(self, rng, state, batch, epoch):
    (image, label) = batch
    x = jnp.array(image, dtype=jnp.float32)
    rng, key = jax.random.split(rng)
    state, loss = train_step(self.model, state, key, x)
    loss = loss.item()
    return state, loss

  def end_of_epoch_callback(self, rng, state, epoch):
    samples = sampling(state, rng, 25)
    fig, axes = plt.subplots(5, 5, figsize=(25, 25))
    for ax, sample in zip(axes.flatten(), samples):
      ax.imshow(sample.squeeze())
      ax.axis('off')
    plt.tight_layout()
    plt.savefig(self.log_path.joinpath(f'samples_epoch_{epoch}.png'))
    plt.close(fig)


@dataclass
class ModelConfig(BaseModelConfig):
  timesteps: int = 1000
  beta_start: float = 0.00085
  beta_end: float = 0.12
  dims: tuple[int] = (64, 128, 256)
  timestep_num: int = 1000
  timestep_dim: int = 256
  channels: int = 1

  def get_model(self):
    return MNISTDiffusion(
      scheduler=Scheduler(num_timesteps=self.timesteps, beta_start=self.beta_start, beta_end=self.beta_end),
      dims=self.dims,
      timestep_num=self.timestep_num,
      timestep_dim=self.timestep_dim,
      channels=self.channels
    )

  def get_trainer(self, log_path):
    return Trainer(self.get_model(), log_path)

  def init(self, model, rng, data0):
    x0, _ = data0
    t0 = jnp.ones((x0.shape[0],), dtype=jnp.int32)
    noise0 = jax.random.normal(rng, x0.shape)
    return model.init(rng, x0, t0, noise0)


@dataclass
class Config(BaseConfig):
  model: ModelConfig = field(default_factory=ModelConfig)


def get_config():
  return Config(experiment_name="unconditional_mnist")
