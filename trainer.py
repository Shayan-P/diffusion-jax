from abc import ABC, abstractmethod


class AbstractTrainer:
    @abstractmethod
    def train_batch(self, rng, state, batch, epoch):
        raise NotImplementedError

    @abstractmethod
    def end_of_epoch_callback(self, rng, state, epoch):
        raise NotImplementedError
