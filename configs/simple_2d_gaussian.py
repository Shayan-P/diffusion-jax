import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import typing as tp

from dataclasses import dataclass, field
from configs.base_conf import BaseConfig, BaseModelConfig, SyncCheckpointerConfig
from model import DDPMConditionalWithMLP, Scheduler
from trainer import AbstractTrainer
from functools import partial
from pathlib import Path


@partial(jax.jit, static_argnames=("data_shape",))
def sampling(state, rng, y, data_shape):
    return state.apply_fn(state.params, method=DDPMConditionalWithMLP.sampling, y=y, data_shape=data_shape, rng=rng)


@partial(jax.jit, static_argnames=("model",))
def train_step(model, state, rng, x, y):
    return model.train_step(state=state, rng=rng, x=x, y=y)


class Trainer(AbstractTrainer):
  def __init__(self, model, x_shape, log_path: Path, log_sample_count: int):
    super().__init__()
    self.model = model
    self.log_path = log_path
    self.x_shape = x_shape
    self.log_sample_count = log_sample_count

    plt.switch_backend('agg') # for headless mode. perhaps remove later do to it's sideeffects

  def train_batch(self, rng, state, batch, epoch):
    (x, y) = batch
    x = jnp.array(x, dtype=jnp.float32)
    y = jnp.array(y, dtype=jnp.float32)
    rng, key = jax.random.split(rng)
    state, loss = train_step(self.model, state, key, x, y=y)
    loss = loss.item()
    return state, loss

  def end_of_epoch_callback(self, rng, state, epoch):
    gen_count = self.log_sample_count
    # this is assuming y is 1D and between 0, 1
    y = jnp.linspace(-3, 3, gen_count)[:, None]
    x = sampling(state, rng, y=y, data_shape=self.x_shape)

    assert x.shape[0] == gen_count
    assert x.shape[1] == 1
    assert len(x.shape) == 2
    x = x.reshape(-1)
    y = y.reshape(-1)
    pts = jnp.stack([x, y], axis=1)

    fig = plt.figure(figsize=(10, 10))
    plt.scatter(pts[:, 0], pts[:, 1]) # later do heat map
    plt.tight_layout()
    plt.savefig(self.log_path.joinpath(f'samples_epoch_{epoch}.png'))
    plt.close(fig)


@dataclass
class ModelConfig(BaseModelConfig):
  timesteps: int = 1000

  dims: tp.Tuple[int] = (64, 128, 256, 1)
  timestep_num: int = 1000
  timestep_dim: int = 128
  # todo we are doing early fusion here... chances are the initial input gets lost in the large timestep dim

  beta_start: float = 0.00085
  beta_end: float = 0.12

  log_sample_count: int = 1000

  def get_model(self):
    return DDPMConditionalWithMLP(
      scheduler=Scheduler(num_timesteps=self.timesteps, beta_start=self.beta_start, beta_end=self.beta_end),
      dims=self.dims,
      timestep_num=self.timestep_num,
      timestep_dim=self.timestep_dim,
    )

  def get_trainer(self, log_path):
    return Trainer(self.get_model(), self.x_shape, log_path, log_sample_count=self.log_sample_count)

  def init(self, model, rng, data0):
    x0, y0 = data0
    self.x_shape = x0.shape[1:]
    t0 = jnp.ones((x0.shape[0],), dtype=jnp.int32)
    noise0 = jax.random.normal(rng, x0.shape)
    return model.init(rng, x0, y0, t0, noise0)


@dataclass
class Config(BaseConfig):
  model: ModelConfig = field(default_factory=ModelConfig)


def get_config():
  config = Config(experiment_name="simple_2d_guassian")
  config.dataset.name = 'gaussian_points'
  config.checkpoint = SyncCheckpointerConfig()
  config.optimizer.start_lr = 1e-3
  config.optimizer.end_lr = 1e-5
  config.optimizer.warmup_steps = 100
  config.dataset.train_batch_size = 1024
  return config
