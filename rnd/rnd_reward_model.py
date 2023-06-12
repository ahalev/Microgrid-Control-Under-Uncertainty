import numpy as np
import torch

from dowel import tabular

from garage import make_optimizer
from garage.torch._functions import zero_optim_grads
from garage.torch.modules import MLPModule

from torch import nn
from torch.utils.data import DataLoader


class RNDModel:
    def __init__(self,
                 obs_dim,
                 output_dim=128,
                 hidden_sizes=(64, 64),
                 batch_size=64,
                 n_train_steps=32,
                 intrinsic_reward_weight=0.01,
                 standardize_intrinsic_reward=True,
                 extrinsic_reward_norm=True,
                 extrinsic_reward_norm_bounds=(0, 1),
                 predictor_optimizer=torch.optim.Adam,
                 predictor_lr=1e-3):

        self.batch_size = batch_size
        self.n_train_steps = n_train_steps
        self.intrinsic_reward_weight = intrinsic_reward_weight
        self.standardize_intrinsic_reward = standardize_intrinsic_reward
        self.extrinsic_reward_norm = extrinsic_reward_norm
        self.extrinsic_reward_norm_bounds = extrinsic_reward_norm_bounds

        self._extrinsic_reward_norm_bound_range = extrinsic_reward_norm_bounds[1] - extrinsic_reward_norm_bounds[0]
        assert self._extrinsic_reward_norm_bound_range > 0, f'Invalid bounds: {extrinsic_reward_norm_bounds}'

        self._reward_model = RNDNetwork(obs_dim, output_dim, hidden_sizes)
        self._reward_model_optimizer = make_optimizer(predictor_optimizer,
                                                      module=self._reward_model.predictor,
                                                      lr=predictor_lr)
        self._reward_running_mean_std = RunningMeanStd()

    def train_once(self, observations):
        dataset = RNDDataset(observations)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        dataloader_iter = iter(dataloader)

        losses = []

        for _ in range(self.n_train_steps):
            loss = self._train_step(obs_minibatch=next(dataloader_iter))
            losses.append(loss)

        self.log_training(losses)
        return losses

    def _train_step(self, obs_minibatch):
        predicted_feature, target_feature = self._reward_model(obs_minibatch)
        mse_loss = nn.MSELoss()
        loss = mse_loss(predicted_feature, target_feature.detach())
        zero_optim_grads(self._reward_model_optimizer)
        loss.backward()
        self._reward_model_optimizer.step()
        return loss.detach().item()

    @staticmethod
    def log_training(losses):
        with tabular.prefix('RNDModel/'):
            tabular.record('AveragePredictorLoss', np.mean(losses))
            tabular.record('MaxPredictorLoss', np.max(losses))

    def transform_rewards(self, obs, extrinsic_rewards):
        intrinsic_rewards = self.compute_intrinsic_rewards(obs)
        transformed_ext_rewards = self._transform_ext_rewards(extrinsic_rewards)
        transformed = transformed_ext_rewards + self.intrinsic_reward_weight * intrinsic_rewards
        assert transformed.shape == extrinsic_rewards.shape

        self.log_rewards(transformed_ext_rewards, intrinsic_rewards, transformed)

        return transformed

    def compute_intrinsic_rewards(self, obs):
        if not isinstance(obs, torch.FloatTensor):
            obs = torch.FloatTensor(obs)

        with torch.no_grad():
            predicted_feature, target_feature = self._reward_model(obs)
            mse_f = nn.MSELoss(reduction='none')
            mse = mse_f(predicted_feature, target_feature).mean(dim=1)
            mse = mse.detach().numpy()

            if self.standardize_intrinsic_reward:
                self._reward_running_mean_std.update(mse)
                mse /= self._reward_running_mean_std.var

            return mse

    def _transform_ext_rewards(self, extrinsic_rewards):
        if self.extrinsic_reward_norm:
            # Transform from self.extrinsic_reward_norm_bounds to (0, 1)
            return (extrinsic_rewards - self.extrinsic_reward_norm_bounds[0]) / self._extrinsic_reward_norm_bound_range

        return extrinsic_rewards

    @staticmethod
    def log_rewards(extrinsic_rewards, intrinsic_rewards, total_rewards):
        with tabular.prefix('RNDRewards/'):

            tabular.record('AverageReward', np.mean(total_rewards))
            tabular.record('AverageAbsReward', np.mean(np.abs(total_rewards)))
            tabular.record('StdReward', np.std(total_rewards))
            tabular.record('MaxReward', np.max(total_rewards))
            tabular.record('MinReward', np.min(total_rewards))

            tabular.record('AverageExtrinsicReward', np.mean(extrinsic_rewards))
            tabular.record('AverageAbsExtrinsicReward', np.mean(np.abs(extrinsic_rewards)))
            tabular.record('StdExtrinsicReward', np.std(extrinsic_rewards))
            tabular.record('MaxExtrinsicReward', np.max(extrinsic_rewards))
            tabular.record('MinExtrinsicReward', np.min(extrinsic_rewards))

            tabular.record('AverageIntrinsicReward', np.mean(intrinsic_rewards))
            tabular.record('StdIntrinsicReward', np.std(intrinsic_rewards))
            tabular.record('MaxIntrinsicReward', np.max(intrinsic_rewards))
            tabular.record('MinIntrinsicReward', np.min(intrinsic_rewards))


class RNDNetwork(nn.Module):
    def __init__(self, obs_dim, output_dim, hidden_sizes):
        super().__init__()
        self.target = MLPModule(obs_dim, output_dim, hidden_sizes)
        self.predictor = MLPModule(obs_dim, output_dim, hidden_sizes)

        for param in self.target.parameters():
            param.requires_grad = False

    def forward(self, obs: torch.Tensor):
        predicted_feature = self.predictor(obs)
        with torch.no_grad():
            target_feature = self.target(obs)
        return predicted_feature, target_feature


class RNDDataset(torch.utils.data.Dataset):
    def __init__(self, obs):
        super().__init__()
        self.obs = torch.FloatTensor(obs)

    def __len__(self):
        return self.obs.shape[0]

    def __getitem__(self, index):
        return self.obs[index]


class RunningMeanStd(object):
    # https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm
    def __init__(self, epsilon=1e-4, shape=()):
        self.mean = np.zeros(shape, 'float64')
        self.var = np.ones(shape, 'float64')
        self.count = epsilon

    def update(self, x):
        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        batch_count = x.shape[0]
        self.update_from_moments(batch_mean, batch_var, batch_count)

    def update_from_moments(self, batch_mean, batch_var, batch_count):
        delta = batch_mean - self.mean
        tot_count = self.count + batch_count

        new_mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * (self.count)
        m_b = batch_var * (batch_count)
        M2 = m_a + m_b + np.square(delta) * self.count * batch_count / (self.count + batch_count)
        new_var = M2 / (self.count + batch_count)

        new_count = batch_count + self.count

        self.mean = new_mean
        self.var = new_var
        self.count = new_count