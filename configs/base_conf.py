import optax
import orbax
import jax
import jax.numpy as jnp
import torch
import dataset as my_dataset

from pathlib import Path
from dataclasses import dataclass, field
from orbax.checkpoint import CheckpointManager
from flax.training import orbax_utils
from utils import EXPERIMENT_PATH
from torchvision import datasets, transforms


@dataclass
class OptimizerConfig:
  start_lr: float = 1e-3
  end_lr: float = 1e-4
  warmup_steps: int = 1000
  decay_steps: int = 5000

  def get_lr_schedule(self):
    return optax.warmup_cosine_decay_schedule(
      init_value=self.start_lr,
      peak_value=self.start_lr,
      warmup_steps=self.warmup_steps,
      decay_steps=self.decay_steps,
      end_value=self.end_lr
    )

  def get_optimizer(self):
    lr_scheduler = self.get_lr_schedule()
    optimizer = optax.adam(lr_scheduler)
    return optimizer


@dataclass
class BaseCheckpointConfig:
  restart_from_last_checkpoint: bool = True
  num_checkpoints: int = 5
  async_timeout_secs: int = 50

  def get_orbax_checkpointer(self):
    raise NotImplementedError
  
  def get_checkpointer(self, experiment_path: Path):
    orbax_checkpointer = self.get_orbax_checkpointer()
    checkpoint_manager: CheckpointManager = orbax.checkpoint.CheckpointManager(
        str(experiment_path.joinpath("checkpoints")),
        orbax_checkpointer,
        orbax.checkpoint.CheckpointManagerOptions(max_to_keep=self.num_checkpoints, create=True))
    return checkpoint_manager

  def save(self, manager: CheckpointManager, checkpoint, idx: int):
    save_args = orbax_utils.save_args_from_target(checkpoint)
    manager.save(idx, checkpoint, save_kwargs={'save_args': save_args})

  def restore_latest_if_can(self, manager: CheckpointManager, checkpoint):
    latest_step = manager.latest_step()
    if self.restart_from_last_checkpoint and latest_step is not None:
      return self.restore(manager, checkpoint, latest_step)
    print("Did not restore from checkpoint")
    return checkpoint

  def restore(self, manager: CheckpointManager, checkpoint, idx: int):
      print(f"loading checkpoint {idx}")
      restore_args = orbax_utils.restore_args_from_target(checkpoint)
      checkpoint = manager.restore(idx, items=checkpoint, restore_kwargs={'restore_args': restore_args})
      return checkpoint


@dataclass
class AsyncCheckpointerConfig(BaseCheckpointConfig):
  async_timeout_secs: int = 50

  def get_orbax_checkpointer(self):
    return orbax.checkpoint.AsyncCheckpointer(orbax.checkpoint.PyTreeCheckpointHandler(), timeout_secs=self.async_timeout_secs)


@dataclass
class SyncCheckpointerConfig(BaseCheckpointConfig):
  def get_orbax_checkpointer(self):
    return orbax.checkpoint.Checkpointer(orbax.checkpoint.PyTreeCheckpointHandler())



@dataclass
class DatasetConfig:
  name: str = "mnist"
  train_batch_size: int = 128

  def get_loader(self):
    if self.name == 'mnist':
      dataset = my_dataset.get_mnist()
    elif self.name == 'circle':
      dataset = my_dataset.get_circle_dataset()
    else:
        raise ValueError(f"Unknown dataset {self.name}")
    # enforce the same batch size for training and testing
    train_loader = my_dataset.NumpyLoader(dataset, batch_size=self.train_batch_size, shuffle=True, drop_last=True)
    return train_loader


@dataclass
class BaseModelConfig:
  def get_model(self):
    raise NotImplementedError
  
  def get_trainer(self, log_path: Path):
    raise NotImplementedError
  
  def init(self, model, rng, data0):
    raise NotImplementedError


@dataclass
class BaseConfig:
  experiment_name: str
  optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
  checkpoint: BaseCheckpointConfig = field(default_factory=AsyncCheckpointerConfig)
  dataset: DatasetConfig = field(default_factory=DatasetConfig)
  seed: int = 42
  epochs: int = 100
  jax_explain_cache_misses: bool = False

  def get_experiment_path(self) -> Path:
    path = EXPERIMENT_PATH.joinpath(self.experiment_name)
    path.mkdir(parents=True, exist_ok=True)
    return path

  def get_log_path(self) -> Path:
    path = self.get_experiment_path().joinpath("logs")
    path.mkdir(exist_ok=True)
    return path
