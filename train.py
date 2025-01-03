import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

from flax.training.train_state import TrainState
from functools import partial
from tqdm import tqdm
from utils import jax_jit_setup
from collections import namedtuple
from absl import app
from ml_collections.config_flags import config_flags
from pprint import pprint
from dataclasses import asdict
from configs import BaseConfig


FLAGS = config_flags.FLAGS
config_flags.DEFINE_config_file('config')


def main(_):
    config: BaseConfig = FLAGS.config

    print("Config: ")
    pprint(config)

    # compilation cache
    jax_jit_setup(config)

    # rng
    rng = jax.random.PRNGKey(config.seed)
    
    # get model
    model = config.model.get_model()

    # data
    train_loader = config.dataset.get_loader()

    # optimizer
    lr_scheduler = config.optimizer.get_lr_schedule()
    optimizer = config.optimizer.get_optimizer()

    # exp path
    experiment_path = config.get_experiment_path()

    # checkpointer
    checkpoint_manager = config.checkpoint.get_checkpointer(experiment_path)

    # log path
    log_path = config.get_log_path()

    # init model
    data0 = next(iter(train_loader))
    params = config.model.init(model, rng, data0)

    state = TrainState.create(
        apply_fn=model.apply,
        params=params,
        tx=optimizer)
    del params

    # checkpoint
    Checkpoint = namedtuple("Checkpoint", ["state", "losses", "epoch", "config"])
    checkpoint = Checkpoint(state=state, losses=jnp.array([], dtype=jnp.float32), epoch=-1, config=asdict(config))
    checkpoint = config.checkpoint.restore_latest_if_can(checkpoint_manager, checkpoint)
    state, losses, epoch = checkpoint.state, checkpoint.losses, checkpoint.epoch
    losses = losses.tolist() if isinstance(losses, jnp.ndarray) else losses

    # get trainer
    trainer = config.model.get_trainer(log_path)

    # start training
    print("Training...")
    for epoch in range(checkpoint.epoch + 1, checkpoint.epoch + 1 + config.epochs):
        print("Epoch: ", epoch)
        with tqdm(train_loader) as pbar:
            for i, batch in enumerate(pbar):
                key, rng = jax.random.split(rng)
                state, loss = trainer.train_batch(key, state, batch, epoch)
                losses.append(loss)
                if i % 10 == 0:
                    pbar.set_description(f"Loss: {loss:.4f}, LR: {lr_scheduler(state.step):.6f}")

        # save checkpoint
        config.checkpoint.save(checkpoint_manager,
                               Checkpoint(state=state, losses=jnp.array(losses), epoch=epoch, config=asdict(config)),
                               idx=epoch)

        # log
        key, rng = jax.random.split(rng)
        trainer.end_of_epoch_callback(key, state, epoch)

    plt.plot(losses)
    plt.savefig(log_path.joinpath('losses.png'))
    print("Training finished")

if __name__ == "__main__":
    app.run(main)
