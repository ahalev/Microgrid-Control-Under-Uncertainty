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
        indices = np.random.permutation(len(batch.actions)-1)
        minibatches = np.array_split(indices, self._minibatches_per_epoch)

        losses = []
        qf_losses, qvals, y_targets = [], [], []
        vf_losses = []
        for minibatch in minibatches:
            loss = self._train_policy_once(batch, minibatch)
            qf_loss, qval, y_target = self._train_qf_once(batch, minibatch)
            vf_loss = self._train_vf_once(batch, minibatch)

            losses.append(loss)

            qf_losses.append(qf_loss)
            qvals.append(qval)
            y_targets.append(y_target)

            vf_losses.append(vf_loss)

        self.update_target_qf()

        self._log_epoch(epoch, qf_losses, qvals, y_targets, vf_losses)

        return losses

    def _obtain_samples(self, trainer, epoch):
        """Obtain samples from self._source.

        Identical to super()._obtain_samples(trainer, epoch), except we return an
        EpisodeBatch instead of a TimeStepBatch for value function training purposes.

        Args:
            trainer (Trainer): Experiment trainer, which may be used to
                obtain samples.
            epoch (int): The current epoch.

        Returns:
            TimeStepBatch: Batch of samples.

        """
        if isinstance(self._source, Policy):
            return super()._obtain_samples(trainer, epoch)
        else:
            batches = []
            while (sum(len(batch.actions)
                       for batch in batches) < self._batch_size):
                batches.append(next(self._source))
            return EpisodeBatch.concatenate(*batches)

    def _train_policy_once(self, batch, minibatch):
        observations = np_to_torch(batch.observations[minibatch])
        actions = np_to_torch(batch.actions[minibatch])
        self._optimizer.zero_grad()
        loss = self._compute_loss(observations, actions)
        loss.backward()
        self._optimizer.step()

        return loss.item()

    def _train_qf_once(self, batch, minibatch):
        if self._qf is None:
            return None, None, None

        observations = np_to_torch(batch.observations[minibatch])
        actions = np_to_torch(batch.actions[minibatch])

        y_target, qval_mask = self._compute_y_target(batch, minibatch)

        # optimize critic
        inputs = observations
        qval = self._qf(inputs, actions)[qval_mask]
        qf_loss = torch.nn.MSELoss()
        qval_loss = qf_loss(qval, y_target)
        zero_optim_grads(self._qf_optimizer)
        qval_loss.backward()
        self._qf_optimizer.step()

        return qval_loss.item(), qval.detach().numpy(), y_target.detach().numpy()

    def _compute_y_target(self, batch, minibatch):
        next_observations = np_to_torch(batch.next_observations[minibatch])
        rewards = batch.rewards[minibatch]
        terminals = batch.terminals[minibatch]

        if self._qf_target == 'learner':
            with torch.no_grad():
                next_actions = self._target_learner(next_observations)
                qval_mask = np.ones(len(minibatch)).astype(bool)

        elif self._qf_target == 'expert':
            next_actions = np_to_torch(batch.actions[minibatch+1])

            # Exclude steps where action was terminal as next_action does not follow
            qval_mask = ~terminals
            next_actions = next_actions[qval_mask]
            next_observations = next_observations[qval_mask]
            rewards = rewards[qval_mask]
            terminals = terminals[qval_mask]
        else:
            raise NameError(self._qf_target)

        rewards = np_to_torch(rewards).reshape(-1, 1)
        terminals = np_to_torch(terminals).reshape(-1, 1)

        with torch.no_grad():
            target_qvals = self._target_qf(next_observations, next_actions)

        y_target = rewards + (1.0 - terminals) * self._discount * target_qvals
        return y_target, qval_mask

    def update_target_qf(self):
        """Update parameters in the target policy and Q-value network."""
        if self._qf is None:
            return

        for t_param, param in zip(self._target_qf.parameters(),
                                  self._qf.parameters()):
            t_param.data.copy_(t_param.data * (1.0 - self._tau) +
                               param.data * self._tau)

        for t_param, param in zip(self._target_learner.parameters(),
                                  self.learner.parameters()):
            t_param.data.copy_(t_param.data * (1.0 - self._tau) +
                               param.data * self._tau)

    def _train_vf_once(self, batch, minibatch):
        if self._value_function is None:
            return None

        returns = np_to_torch(
            np.stack([
                discount_cumsum(reward, self._discount)
                for reward in batch.padded_rewards
            ]))

        valids = batch.lengths
        obs_flat = np_to_torch(batch.observations)
        returns_flat = torch.cat(filter_valids(returns, valids))

        zero_optim_grads(self._vf_optimizer)
        loss = self._value_function.compute_loss(obs_flat[minibatch], returns_flat[minibatch])
        loss.backward()
        self._vf_optimizer.step()

        return loss.item()

    def _log_epoch(self, epoch, qf_losses, qvals, y_targets, vf_losses):
        tabular.record('Epoch', epoch)
        if self._qf is not None:
            qvals = np.concatenate([x.reshape(-1) for x in qvals])
            y_targets = np.concatenate([x.reshape(-1) for x in y_targets])

            tabular.record('QFunction/AverageQFunctionLoss',
                           np.mean(qf_losses))
            tabular.record('QFunction/AverageQ', np.mean(qvals))
            tabular.record('QFunction/MaxQ', np.max(qvals))
            tabular.record('QFunction/AverageAbsQ',
                           np.mean(np.abs(qvals)))
            tabular.record('QFunction/AverageY', np.mean(y_targets))
            tabular.record('QFunction/MaxY', np.max(y_targets))
            tabular.record('QFunction/AverageAbsY',
                           np.mean(np.abs(y_targets)))

        if self._value_function is not None:
            with tabular.prefix(self._value_function.name):
                tabular.record('/AverageValueFunctionLoss', np.mean(vf_losses))

