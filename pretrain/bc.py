import torch

from garage import _Default
from garage.torch.algos import BC as _GarageBC


class BC(_GarageBC):
    def __init__(
            self,
            env_spec,
            learner,
            *,
            batch_size,
            source=None,
            sampler=None,
            policy_optimizer=torch.optim.Adam,
            policy_lr=_Default(1e-3),
            loss='log_prob',
            minibatches_per_epoch=16,
            name='BC',
            qf=None
    ):
        super().__init__(
            env_spec,
            learner,
            batch_size=batch_size,
            source=source,
            sampler=sampler,
            policy_optimizer=policy_optimizer,
            policy_lr=policy_lr,
            loss=loss,
            minibatches_per_epoch=minibatches_per_epoch,
            name=name
        )

        self._qf = qf
