import jax.numpy as jnp
import numpy as np

from torchvision import datasets, transforms
from jax.tree_util import tree_map
from torch.utils import data
from torchvision.datasets import MNIST
from utils import DATASET_PATH

def numpy_collate(batch):
  return tree_map(np.asarray, data.default_collate(batch))

class NumpyLoader(data.DataLoader):
  def __init__(self, dataset, batch_size=1,
                shuffle=False, sampler=None,
                batch_sampler=None, num_workers=0,
                pin_memory=False, drop_last=False,
                timeout=0, worker_init_fn=None):
    super(self.__class__, self).__init__(dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
        batch_sampler=batch_sampler,
        num_workers=num_workers,
        collate_fn=numpy_collate,
        pin_memory=pin_memory,
        drop_last=drop_last,
        timeout=timeout,
        worker_init_fn=worker_init_fn)


def get_mnist():
    def transform(pic):
        return np.array(pic, dtype=jnp.float32).reshape(28, 28, 1)
    return datasets.MNIST(
       str(DATASET_PATH.joinpath("mnist").absolute()),
       download=True, transform=transform)


def get_circle_dataset():
  theta = np.linspace(0, 2 * np.pi, 50000)
  x = np.cos(theta)
  y = np.sin(theta)
  return [
     (np.array([x]), np.array([y]))
     for x, y in zip(x, y)
  ]
