import torch
import numpy as np

from copy import deepcopy
from dowel import tabular

from garage import _Default, make_optimizer, EpisodeBatch
from garage.np.policies import Policy
from garage.np import discount_cumsum
from garage.torch import np_to_torch, filter_valids
from garage.torch._functions import zero_optim_grads
from garage.torch.algos import BC as _GarageBC



# TODO need to use this, figure out how to initialize qf (and value function for ppo)

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
            qf=None,
            qf_optimizer=torch.optim.Adam,
            qf_lr=_Default(1e-3),
            qf_target='expert',
            target_update_tau=0.01,
            value_function=None,
            vf_optimizer=torch.optim.Adam,
            vf_lr=_Default(1e-3),
            discount=0.99,
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
        if self._qf:
            self._qf_optimizer = make_optimizer(qf_optimizer, module=self._qf, lr=qf_lr) if self._qf else None
        else:
            self._qf_optimizer = None

        assert qf_target in ('learner', 'expert'), "qf_target must be 'learner' or 'expert'"

        self._qf_target = qf_target
        self._tau = target_update_tau
        self._discount = discount

        self._target_learner = deepcopy(self.learner)
        self._target_qf = deepcopy(self._qf)

        self._value_function = value_function

        if self._value_function:
            self._vf_optimizer = make_optimizer(vf_optimizer, module=self._value_function, lr=vf_lr)
        else:
            self._vf_optimizer = None

    def _train_once(self, trainer, epoch):
        """Obtain samplers and train for one epoch.

        Args:
            trainer (Trainer): Experiment trainer, which may be used to
                obtain samples.
            epoch (int): The current epoch.

        Returns:
            List[float]: Losses.

        """
        if self._qf is None:
            return super()._train_once(trainer, epoch)

        batch = self._obtain_samples(trainer, epoch)
        indices = np.random.permutation(len(batch.actions))
        minibatches = np.array_split(indices, self._minibatches_per_epoch)
        losses = []
        for minibatch in minibatches:
            observations = np_to_torch(batch.observations[minibatch])
            actions = np_to_torch(batch.actions[minibatch])
            self._optimizer.zero_grad()
            loss = self._compute_loss(observations, actions)
            loss.backward()
            losses.append(loss.item())
            self._optimizer.step()
        return losses
