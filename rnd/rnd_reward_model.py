import numpy as np
import torch

from abc import abstractmethod
from dowel import tabular

from garage import make_optimizer
from garage.torch._functions import zero_optim_grads
from garage.torch.modules import MLPModule

from torch import nn
from torch.utils.data import DataLoader


EPS = 1e-6


class RNDBase:
    def __init__(self):
        self.external_reward_running_mean_std = RunningMeanStd()

    @abstractmethod
    def train_once(self, observations):
        pass

    @abstractmethod
    def transform_rewards(self, obs, extrinsic_rewards):
        pass

    def _transform_ext_rewards(self, extrinsic_rewards):
        # Transform from self.extrinsic_reward_norm_bounds to (0, 1)
        self.external_reward_running_mean_std.update(extrinsic_rewards)
        return (extrinsic_rewards - self.external_reward_running_mean_std.mean) / self.external_reward_running_mean_std.var


class StandardizeExternal(RNDBase):

    def train_once(self, observations):
        pass

    def transform_rewards(self, obs, extrinsic_rewards):
        return self._transform_ext_rewards(extrinsic_rewards)


class RNDModel(RNDBase):
    def __init__(self,
                 obs_dim,
                 output_dim=128,
                 hidden_sizes=(64, 64),
                 batch_size=64,
                 n_train_steps=32,
                 intrinsic_reward_weight=0.5,
                 bound_reward_weight=None,
                 bound_reward_weight_transient_epochs=0,
                 bound_reward_weight_initial_ratio=1-EPS,
                 max_epochs=100,
                 standardize_intrinsic_reward=True,
                 standardize_extrinsic_reward=True,
                 predictor_optimizer=torch.optim.Adam,
                 predictor_lr=1e-3,
                 **mlp_kwargs):

        self.batch_size = batch_size
        self.n_train_steps = n_train_steps

        self.intrinsic_reward_weight_val = intrinsic_reward_weight
        self.bound_reward_weight = bound_reward_weight
        self.bound_reward_weight_transient_epochs = bound_reward_weight_transient_epochs
        self.bound_reward_weight_initial_ratio = bound_reward_weight_initial_ratio
        self.max_epochs = max_epochs

        self.standardize_intrinsic_reward = standardize_intrinsic_reward
        self.standardize_extrinsic_reward = standardize_extrinsic_reward

        self._reward_model = RNDNetwork(obs_dim, output_dim, hidden_sizes, **mlp_kwargs)
        self._reward_model_optimizer = make_optimizer(predictor_optimizer,
                                                      module=self._reward_model.predictor,
                                                      lr=predictor_lr)

        self._intrinsic_reward_running_mean_std = RunningMeanStd()

        self._epoch = 0

        super().__init__()

    def intrinsic_reward_weight(self, standardized_external_rewards, standardized_internal_rewards):
        mean_reward_ratio = np.abs(standardized_external_rewards).mean() / np.abs(standardized_internal_rewards).mean()

        def _get_info(int_reward_weight, computed_int_reward_weight, int_ratio_bound):
            keys = 'IntrinsicRewardWeight', 'ComputedIntrinsicRewardWeight', 'IntrinsicRatioBound', 'MeanExtIntAbsRewardRatio'
            vals = [int_reward_weight, computed_int_reward_weight, int_ratio_bound, mean_reward_ratio]

            return dict(zip(keys, vals))

        if not self.bound_reward_weight or self._epoch < self.bound_reward_weight_transient_epochs-1:
            info = _get_info(self.intrinsic_reward_weight_val, self.intrinsic_reward_weight_val, np.nan)
            return self.intrinsic_reward_weight_val, info

        elif self.bound_reward_weight == 'cosine':
            intrinsic_ratio_bound = self._cosine_int_ratio_bound(self._epoch-self.bound_reward_weight_transient_epochs)
        else:
            raise ValueError(self.bound_reward_weight)

        intrinsic_ratio_bound = np.clip(intrinsic_ratio_bound, EPS, 1-EPS)
        computed_int_reward_weight = intrinsic_ratio_bound * mean_reward_ratio / (1 - intrinsic_ratio_bound)
        int_reward_weight = min(computed_int_reward_weight, self.intrinsic_reward_weight_val)

        info = _get_info(int_reward_weight, computed_int_reward_weight, intrinsic_ratio_bound)
        return int_reward_weight, info

    def _cosine_int_ratio_bound(self, epoch):
        ratio_max = self.bound_reward_weight_initial_ratio
        ratio_min = self.intrinsic_reward_weight_val

        return ratio_min + 0.5 * (ratio_max - ratio_min) * (1 + np.cos(np.pi * epoch / self.max_epochs))

    def train_once(self, observations):
        self._epoch += 1

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
        if self.standardize_extrinsic_reward:
            transformed_ext_rewards = self._transform_ext_rewards(extrinsic_rewards)
        else:
            transformed_ext_rewards = extrinsic_rewards

        intrinsic_rewards = self.compute_intrinsic_rewards(obs)
        intrinsic_reward_weight, reward_weight_info = self.intrinsic_reward_weight(transformed_ext_rewards, intrinsic_rewards)

        transformed = transformed_ext_rewards + intrinsic_reward_weight * intrinsic_rewards
        assert transformed.shape == extrinsic_rewards.shape

        self.log_rewards(extrinsic_rewards, intrinsic_rewards, transformed_ext_rewards,
                         transformed, reward_weight_info)

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
                self._intrinsic_reward_running_mean_std.update(mse)
                mse /= self._intrinsic_reward_running_mean_std.var ** (1/2)

            return mse

    def log_rewards(self, extrinsic_rewards, intrinsic_rewards, transformed_extrinsic_rewards,
                    total_rewards, reward_weight_info):

        trans_ext_abs = np.abs(transformed_extrinsic_rewards)
        int_abs = np.abs(intrinsic_rewards*reward_weight_info['IntrinsicRewardWeight'])

        ratios_of_overall = {
            'RNDRewardsExtrinsicTransformed/AveragePercentOfOverallAbs': np.mean(trans_ext_abs/(trans_ext_abs+int_abs)),
            'RNDRewardsIntrinsic/AveragePercentOfOverallAbs': np.mean(int_abs/(trans_ext_abs+int_abs))
        }

        tabular.record_many(ratios_of_overall)

        with tabular.prefix('RNDRewardsOverall/'):
            tabular.record_many(reward_weight_info)

        for k, rewards in dict(
                Overall=total_rewards,
                ExtrinsicTransformed=transformed_extrinsic_rewards,
                Extrinsic=extrinsic_rewards,
                Intrinsic=intrinsic_rewards
        ).items():

            with tabular.prefix(f'RNDRewards{k}/'):
                tabular.record('AverageReward', np.mean(rewards))
                tabular.record('AverageAbsReward', np.mean(np.abs(rewards)))
                tabular.record('StdReward', np.std(rewards))
                tabular.record('MaxReward', np.max(rewards))
                tabular.record('MinReward', np.min(rewards))

        if self.standardize_extrinsic_reward:
            with tabular.prefix(f'RNDRewardsExtrinsic/'):
                tabular.record('RewardRunningMean', self.external_reward_running_mean_std.mean)
                tabular.record('RewardRunningVar', self.external_reward_running_mean_std.var)

        if self.standardize_intrinsic_reward:
            with tabular.prefix(f'RNDRewardsIntrinsic/'):
                tabular.record('RewardRunningMean', self._intrinsic_reward_running_mean_std.mean)
                tabular.record('RewardRunningVar', self._intrinsic_reward_running_mean_std.var)


class RNDNetwork(nn.Module):
    def __init__(self, obs_dim, output_dim, hidden_sizes, **mlp_kwargs):
        super().__init__()

        self.target = MLPModule(obs_dim, output_dim, hidden_sizes, **mlp_kwargs)
        self.predictor = MLPModule(obs_dim, output_dim, hidden_sizes, **mlp_kwargs)

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


class RunningMeanStd:
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
