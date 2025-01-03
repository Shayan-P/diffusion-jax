import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

from dataclasses import dataclass, field
from configs.base_conf import BaseConfig, BaseModelConfig
from model import DDPMConditional, Scheduler
from trainer import AbstractTrainer
from functools import partial
from pathlib import Path


@partial(jax.jit, static_argnames=("data_shape",))
def sampling(state, rng, labels, data_shape):
    return state.apply_fn(state.params, method=DDPMConditional.sampling, labels=labels, data_shape=data_shape, rng=rng)


@partial(jax.jit, static_argnames=("model",))
def train_step(model, state, rng, x, labels):
    return model.train_step(state=state, rng=rng, x=x, labels=labels)


class Trainer(AbstractTrainer):
  def __init__(self, model, image_shape, log_path: Path):
    super().__init__()
    self.model = model
    self.log_path = log_path
    self.image_shape = image_shape

  def train_batch(self, rng, state, batch, epoch):
    (images, labels) = batch
    x = jnp.array(images, dtype=jnp.float32)
    labels = jnp.array(labels, dtype=jnp.int32)
    rng, key = jax.random.split(rng)
    state, loss = train_step(self.model, state, key, x, labels=labels)
    loss = loss.item()
    return state, loss

  def end_of_epoch_callback(self, rng, state, epoch):
    label_count = 10
    from_each = 5
    labels = jnp.arange(label_count)[:, None].repeat(from_each, axis=1).swapaxes(0, 1).reshape(-1)
    samples = sampling(state, rng, labels=labels, data_shape=self.image_shape)

    fig, axes = plt.subplots(from_each, label_count, figsize=(50, 25))
    for ax, sample in zip(axes.flatten(), samples):
      ax.imshow(sample.squeeze())
      ax.axis('off')
    plt.tight_layout()
    plt.savefig(self.log_path.joinpath(f'samples_epoch_{epoch}.png'))
    plt.close(fig)


@dataclass
class ModelConfig(BaseModelConfig):
  label_count: int = 10
  label_dim: int = 64
  timesteps: int = 1000
  beta_start: float = 0.00085
  beta_end: float = 0.12
  dims: tuple[int] = (64, 128, 256)
  timestep_num: int = 1000
  timestep_dim: int = 256
  channels: int = 1

  def get_model(self):
    return DDPMConditional(
      scheduler=Scheduler(num_timesteps=self.timesteps, beta_start=self.beta_start, beta_end=self.beta_end),
      dims=self.dims,
      timestep_num=self.timestep_num,
      timestep_dim=self.timestep_dim,
      channels=self.channels,
      label_count=self.label_count,
      label_dim=self.label_dim
    )

  def get_trainer(self, log_path):
    return Trainer(self.get_model(), self.image_shape, log_path)

  def init(self, model, rng, data0):
    x0, labels0 = data0
    self.image_shape = x0.shape[1:]
    t0 = jnp.ones((x0.shape[0],), dtype=jnp.int32)
    noise0 = jax.random.normal(rng, x0.shape)
    return model.init(rng, x0, labels0, t0, noise0)


@dataclass
class Config(BaseConfig):
  model: ModelConfig = field(default_factory=ModelConfig)


def get_config():
  config = Config(experiment_name="conditional_mnist")
  config.dataset.name = 'mnist'
  # todo change the label_count and dim for other datases...
  return config
