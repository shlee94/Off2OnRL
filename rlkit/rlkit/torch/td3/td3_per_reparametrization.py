from collections import OrderedDict, namedtuple
from typing import Tuple

import numpy as np
import math
import torch
import torch.optim as optim
from rlkit.core.loss import LossFunction, LossStatistics
from torch import nn as nn

from rlkit.launchers import conf

import rlkit.torch.pytorch_util as ptu
from rlkit.core.eval_util import create_stats_ordered_dict
from rlkit.torch.torch_rl_algorithm import TorchTrainer
from rlkit.core.logging import add_prefix
import gtimer as gt


def update_mean_var_count_from_moments(
    mean, var, count, batch_mean, batch_var, batch_count
):
    delta = batch_mean - mean
    tot_count = count + batch_count
    new_mean = mean + delta + batch_count / tot_count
    m_a = var * count
    m_b = batch_var * batch_count
    M2 = m_a + m_b + torch.pow(delta, 2) * count * batch_count / tot_count
    new_var = M2 / tot_count
    new_count = tot_count
    return new_mean, new_var, new_count


class TorchRunningMeanStd:
    def __init__(self, epsilon=1e-4, shape=(), device=None):
        self.mean = torch.zeros(shape, device=device)
        self.var = torch.ones(shape, device=device)
        self.count = epsilon

    def update(self, x):
        with torch.no_grad():
            batch_mean = torch.mean(x, dim=0)
            batch_var = torch.var(x, dim=0)
            batch_count = x.shape[0]
            self.update_from_moments(batch_mean, batch_var, batch_count)

    def update_from_moments(self, batch_mean, batch_var, batch_count):
        self.mean, self.var, self.count = update_mean_var_count_from_moments(
            self.mean, self.var, self.count, batch_mean, batch_var, batch_count
        )

    @property
    def std(self):
        return torch.sqrt(self.var)


class TD3PERReparamTrainer(TorchTrainer):
    def __init__(
        self,
        env,
        policy,
        target_policy,
        ft_init_alpha,
        qf1,
        qf2,
        target_qf1,
        target_qf2,
        weight_net,
        model_dir,
        priority_replay_buffer,
        offline_dataset_size,  # for logging
        normalization_type,
        w_activation_type,
        self_norm,
        fix_scale_parameters,
        target_policy_noise=0.2,
        target_policy_noise_clip=0.5,
        sweep_interval=None,
        temperature=5.0,
        discount=0.99,
        reward_scale=1.0,
        policy_lr=1e-3,
        qf_lr=1e-3,
        weight_net_lr=3e-4,
        optimizer_class=optim.Adam,
        soft_target_tau=1e-2,
        target_update_period=1,
        plotter=None,
        render_eval_paths=False,
        use_automatic_entropy_tuning=True,
        target_entropy=None,
        deterministic_backup=False,
        separate_buffers=False,
        save_checkpoints=False,
    ):
        super().__init__()
        self.env = env
        self.policy = policy
        self.target_policy = target_policy
        self.qf1 = qf1
        self.qf2 = qf2
        self.target_qf1 = target_qf1
        self.target_qf2 = target_qf2
        self.weight_net = weight_net
        self.soft_target_tau = soft_target_tau
        self.target_update_period = target_update_period
        self.priority_replay_buffer = priority_replay_buffer
        self.self_norm = self_norm
        self.offline_dataset_size = offline_dataset_size

        self.target_policy_noise = target_policy_noise
        self.target_policy_noise_clip = target_policy_noise_clip

        if w_activation_type == "relu":
            self.w_activation = lambda x: torch.relu(x)
        elif w_activation_type == "exp":
            self.w_activation = lambda x: torch.exp(x)

        self.temperature = temperature

        self.normalization_type = normalization_type
        if normalization_type == "running_mean":
            self.weight_normalizer = TorchRunningMeanStd(shape=(1,), device=ptu.device)
            self.sweep_interval = sweep_interval
        elif normalization_type == "minibatch":
            pass

        self.save_checkpoints = save_checkpoints

        self.plotter = plotter
        self.render_eval_paths = render_eval_paths

        self.qf_criterion = nn.MSELoss(reduction="none")
        self.vf_criterion = nn.MSELoss()

        self.policy_optimizer = optimizer_class(
            self.policy.parameters(),
            lr=policy_lr,
        )
        self.weight_optimizer = optimizer_class(
            self.weight_net.parameters(),
            lr=weight_net_lr,
        )
        if fix_scale_parameters:
            self.qf1_optimizer = optimizer_class(
                [
                    param
                    for name, param in self.qf1.named_parameters()
                    if "rescale" not in name
                ],
                lr=qf_lr,
            )
            self.qf2_optimizer = optimizer_class(
                [
                    param
                    for name, param in self.qf2.named_parameters()
                    if "rescale" not in name
                ],
                lr=qf_lr,
            )
        else:
            self.qf1_optimizer = optimizer_class(
                self.qf1.parameters(),
                lr=qf_lr,
            )
            self.qf2_optimizer = optimizer_class(
                self.qf2.parameters(),
                lr=qf_lr,
            )

        self.discount = discount
        self.reward_scale = reward_scale
        self._n_train_steps_total = 0
        self._need_to_update_eval_statistics = True
        self.eval_statistics = OrderedDict()

        self.model_dir = conf.LOCAL_LOG_DIR + "/" + model_dir

        self.deterministic_backup = deterministic_backup

    def train_from_torch(self, batch):
        gt.blank_stamp()

        """ Update Weights """
        offline_obs = batch["offline_observations"]
        offline_actions = batch["offline_actions"]

        online_obs = batch["online_observations"]
        online_actions = batch["online_actions"]

        # weight network loss calculation!
        offline_weight = self.w_activation(
            self.weight_net(offline_obs, offline_actions)
        )

        offline_f_star = -torch.log(2.0 / (offline_weight + 1) + 1e-10)

        online_weight = self.w_activation(self.weight_net(online_obs, online_actions))
        online_f_prime = torch.log(2 * online_weight / (online_weight + 1) + 1e-10)

        weight_loss = (offline_f_star - online_f_prime).mean()

        offline_samples_ratio = (
            (batch["idxs"] < self.offline_dataset_size).float().mean()
        )
        online_samples_ratio = (
            (batch["idxs"] >= self.offline_dataset_size).float().mean()
        )

        self.weight_optimizer.zero_grad()
        weight_loss.backward()
        self.weight_optimizer.step()

        """
        Update critics
        """

        rewards = batch["full_rewards"]
        terminals = batch["full_terminals"]
        obs = batch["full_observations"]
        actions = batch["full_actions"]
        next_obs = batch["full_next_observations"]
        batch_size = obs.shape[0]

        next_actions = self.target_policy(next_obs).normal_mean
        noise = ptu.randn(next_actions.shape) * self.target_policy_noise
        noise = torch.clamp(
            noise, -self.target_policy_noise_clip, self.target_policy_noise_clip
        ).to(ptu.device)
        noisy_next_actions = next_actions + noise

        target_q1_values = self.target_qf1(next_obs, noisy_next_actions)
        target_q2_values = self.target_qf2(next_obs, noisy_next_actions)
        target_q_values = torch.min(target_q1_values, target_q2_values)
        q_target = (
            self.reward_scale * rewards
            + (1.0 - terminals) * self.discount * target_q_values
        )
        q_target = q_target.detach()

        q1_pred = self.qf1(obs, actions)
        bellman_errors_1 = (q1_pred - q_target) ** 2
        qf1_loss = bellman_errors_1.mean()

        q2_pred = self.qf2(obs, actions)
        bellman_errors_2 = (q2_pred - q_target) ** 2
        qf2_loss = bellman_errors_2.mean()

        self.qf1_optimizer.zero_grad()
        qf1_loss.backward()
        self.qf1_optimizer.step()

        self.qf2_optimizer.zero_grad()
        qf2_loss.backward()
        self.qf2_optimizer.step()

        with torch.no_grad():
            weight = self.w_activation(self.weight_net(obs, actions))
            if self.normalization_type == "running_mean":
                self.weight_normalizer.update(weight ** (self.temperature) + 1e-10)
            elif self.normalization_type == "minibatch":
                normalized_weight = (weight ** (1 / self.temperature)) / (
                    (offline_weight ** (1 / self.temperature)).mean() + 1e-10
                )

        """ Update policy """
        if self._n_train_steps_total % self.target_update_period == 0:
            policy_actions = self.policy(obs).normal_mean
            q_output = self.qf1(obs, policy_actions)
            policy_loss = -q_output.mean()

            self.policy_optimizer.zero_grad()
            policy_loss.backward()
            self.policy_optimizer.step()

            ptu.soft_update_from_to(self.qf1, self.target_qf1, self.soft_target_tau)
            ptu.soft_update_from_to(self.qf2, self.target_qf2, self.soft_target_tau)
            ptu.soft_update_from_to(
                self.policy, self.target_policy, self.soft_target_tau
            )

        self._n_train_steps_total += 1

        if self._need_to_update_eval_statistics:
            eval_statistics = OrderedDict()

            eval_statistics["Onlines Samples Ratio"] = np.mean(
                ptu.get_numpy(online_samples_ratio)
            )
            eval_statistics["Offline Samples Ratio"] = np.mean(
                ptu.get_numpy(offline_samples_ratio)
            )

            eval_statistics["QF1 Loss"] = np.mean(ptu.get_numpy(qf1_loss))
            eval_statistics["QF2 Loss"] = np.mean(ptu.get_numpy(qf2_loss))
            eval_statistics["Policy Loss"] = np.mean(ptu.get_numpy(policy_loss))
            eval_statistics["Weight Loss"] = np.mean(ptu.get_numpy(weight_loss))

            # eval_statistics["Q1 Scale"] = ptu.get_numpy(
            #     self.qf1.log_rescale_coeff.exp()
            # )
            # eval_statistics["Q2 Scale"] = ptu.get_numpy(
            #     self.qf2.log_rescale_coeff.exp()
            # )
            # eval_statistics["Q1 Bias"] = ptu.get_numpy(self.qf1.rescale_bias)
            # eval_statistics["Q2 Bias"] = ptu.get_numpy(self.qf2.rescale_bias)

            eval_statistics.update(
                create_stats_ordered_dict(
                    "Offline Weights", ptu.get_numpy(offline_weight)
                )
            )
            eval_statistics.update(
                create_stats_ordered_dict(
                    "Online Weights", ptu.get_numpy(online_weight)
                )
            )
            eval_statistics.update(
                create_stats_ordered_dict("Full Weights", ptu.get_numpy(weight))
            )
            eval_statistics.update(
                create_stats_ordered_dict(
                    "Q1 Predictions",
                    ptu.get_numpy(q1_pred),
                )
            )
            eval_statistics.update(
                create_stats_ordered_dict(
                    "Q2 Predictions",
                    ptu.get_numpy(q2_pred),
                )
            )
            eval_statistics.update(
                create_stats_ordered_dict(
                    "Q Targets",
                    ptu.get_numpy(q_target),
                )
            )

            self.eval_statistics = eval_statistics

            # Compute statistics using only one batch per epoch
            self._need_to_update_eval_statistics = False
        gt.stamp("sac training", unique=False)

        if self._n_train_steps_total % 25000 == 0 and self.save_checkpoints:
            self.save(self._n_train_steps_total)

        # Sweep
        if (
            self.normalization_type == "running_mean"
            and self.sweep_interval is not None
            and self._n_train_steps_total % self.sweep_interval == 0
        ):
            self.sweep()

    def sweep(self):
        buffer_size = self.priority_replay_buffer._size
        indices = np.random.randn(buffer_size).argsort()
        batch_size = 2048
        steps_per_epoch = int(math.ceil(buffer_size / batch_size))
        for i in range(steps_per_epoch):
            idx = indices[i * batch_size : (i + 1) * batch_size]
            obs = torch.tensor(
                self.priority_replay_buffer.transitions._observations[idx]
            ).ptu(device)
            acts = torch.tensor(
                self.priority_replay_buffer.transitions._observations[idx]
            ).ptu(device)
            tree_idx = idx + self.priority_replay_buffer.transitions.tree_start
            with torch.no_grad():
                weights = (
                    self.weight_net(obs, acts).squeeze().detach().cpu().numpy()
                ) / self.weight_normalizer.mean  ### !!! NORMALIZE
            self.priority_replay_buffer.update_priorities(idx, weights + 1e-10)

    def get_diagnostics(self):
        stats = super().get_diagnostics()
        stats.update(self.eval_statistics)
        return stats

    def end_epoch(self, epoch):
        self._need_to_update_eval_statistics = True

    def save(self, step):
        torch.save(self.policy.state_dict(), "%sactor_%s.pt" % (self.model_dir, step))
        torch.save(self.qf1.state_dict(), "%scritic1_%s.pt" % (self.model_dir, step))
        torch.save(self.qf2.state_dict(), "%scritic2_%s.pt" % (self.model_dir, step))
        torch.save(
            self.target_qf1.state_dict(),
            "%starget_critic1_%s.pt" % (self.model_dir, step),
        )
        torch.save(
            self.target_qf2.state_dict(),
            "%starget_critic2_%s.pt" % (self.model_dir, step),
        )
        torch.save(
            self.weight_net.state_dict(), "%sweight_%s.pt" % (self.model_dir, step)
        )

    @property
    def networks(self):
        return [
            self.policy,
            self.qf1,
            self.qf2,
            self.target_qf1,
            self.target_qf2,
            self.weight_net,
        ]

    @property
    def optimizers(self):
        return [
            self.qf1_optimizer,
            self.qf2_optimizer,
            self.policy_optimizer,
            self.weight_optimizer,
        ]

    def get_snapshot(self):
        return dict(
            policy=self.policy,
            target_policy=self.target_policy,
            qf1=self.qf1,
            qf2=self.qf2,
            target_qf1=self.target_qf1,
            target_qf2=self.target_qf2,
            weight_net=self.weight_net,
        )
