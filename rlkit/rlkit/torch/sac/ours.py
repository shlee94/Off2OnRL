from collections import OrderedDict, namedtuple
from typing import Tuple

import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
from rlkit.core.loss import LossFunction, LossStatistics
from torch import nn as nn

from rlkit.launchers import conf

import rlkit.torch.pytorch_util as ptu
from rlkit.core.eval_util import create_stats_ordered_dict
from rlkit.torch.torch_rl_algorithm import TorchTrainer
from rlkit.core.logging import add_prefix
import gtimer as gt

SACLosses = namedtuple(
    "SACLosses",
    "policy_loss qf1_loss qf2_loss alpha_loss weight_loss",
)

class OursTrainer(TorchTrainer, LossFunction):
    def __init__(
        self,
        env,
        policy,
        qf1,
        qf2,
        target_qf1,
        target_qf2,
        weight_net,
        priority_replay_buffer,
        offline_dataset_size,
        init_alpha=1.0,
        alpha_lr=3e-4,
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
        ensemble_size=5,
    ):
        super().__init__()
        self.env = env
        self.policy = policy
        self.qf1 = qf1
        self.qf2 = qf2
        self.target_qf1 = target_qf1
        self.target_qf2 = target_qf2
        self.weight_net = weight_net
        self.soft_target_tau = soft_target_tau
        self.target_update_period = target_update_period

        self.priority_replay_buffer = priority_replay_buffer
        self.offline_dataset_size = offline_dataset_size

        self.w_activation = lambda x: torch.relu(x)

        self.temperature = temperature

        self.use_automatic_entropy_tuning = use_automatic_entropy_tuning
        if self.use_automatic_entropy_tuning:
            if target_entropy is None:
                # Use heuristic value from SAC paper
                self.target_entropy = -np.prod(self.env.action_space.shape).item()
            else:
                self.target_entropy = target_entropy
            # self.log_alpha = ptu.zeros(1, requires_grad=True)
            self.log_alpha = ptu.tensor(np.log(init_alpha), requires_grad=True)

            self.alpha_optimizer = optimizer_class(
                [self.log_alpha],
                lr=alpha_lr,
            )

        self.plotter = plotter
        self.render_eval_paths = render_eval_paths

        self.qf_criterion = nn.MSELoss(reduction="none")
        self.vf_criterion = nn.MSELoss()

        self.policy_optimizer = optimizer_class(self.policy.parameters(), lr=policy_lr)
        self.qf1_optimizer = optimizer_class(
            self.qf1.parameters(),
            lr=qf_lr,
        )
        self.qf2_optimizer = optimizer_class(
            self.qf2.parameters(),
            lr=qf_lr,
        )
        self.weight_optimizer = optimizer_class(
            self.weight_net.parameters(),
            lr=weight_net_lr,
        )
        self.discount = discount
        self.reward_scale = reward_scale
        self._n_train_steps_total = 0
        self._need_to_update_eval_statistics = True
        self.eval_statistics = OrderedDict()

        self.ensemble_size = ensemble_size


    def train_from_torch(self, batch):
        gt.blank_stamp()
        losses, stats = self.compute_loss(
            batch,
            skip_statistics=not self._need_to_update_eval_statistics,
        )
        """
        Update networks
        """
        if self.use_automatic_entropy_tuning:
            self.alpha_optimizer.zero_grad()
            losses.alpha_loss.backward()
            self.alpha_optimizer.step()

        self.policy_optimizer.zero_grad()
        losses.policy_loss.backward()
        self.policy_optimizer.step()

        self.qf1_optimizer.zero_grad()
        losses.qf1_loss.backward()
        self.qf1_optimizer.step()

        self.qf2_optimizer.zero_grad()
        losses.qf2_loss.backward()
        self.qf2_optimizer.step()

        self.weight_optimizer.zero_grad()
        losses.weight_loss.backward()
        self.weight_optimizer.step()

        self._n_train_steps_total += 1

        self.try_update_target_networks()
        if self._need_to_update_eval_statistics:
            self.eval_statistics = stats
            # Compute statistics using only one batch per epoch
            self._need_to_update_eval_statistics = False
        gt.stamp("sac training", unique=False)


    def try_update_target_networks(self):
        if self._n_train_steps_total % self.target_update_period == 0:
            self.update_target_networks()

    def update_target_networks(self):
        ptu.soft_update_from_to(self.qf1, self.target_qf1, self.soft_target_tau)
        ptu.soft_update_from_to(self.qf2, self.target_qf2, self.soft_target_tau)

    def sweep(self):
        pass

    def compute_loss(
        self,
        batch,
        skip_statistics=False,
    ) -> Tuple[SACLosses, LossStatistics]:

        """
        IS Weight network
        """
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

        """
        RL Training
        """
        obs = batch["rl_observations"]
        actions = batch["rl_actions"]
        next_obs = batch["rl_next_observations"]
        rewards = batch["rl_rewards"]
        terminals = batch["rl_terminals"]

        """
        Policy and Alpha Loss
        """

        dist = self.policy(obs)
        new_obs_actions, log_pi = dist.rsample_and_logprob()
        log_pi = log_pi.unsqueeze(-1)

        if self.use_automatic_entropy_tuning:
            alpha_loss = -(
                self.log_alpha * (log_pi + self.target_entropy).detach()
            ).mean()
            alpha = self.log_alpha.exp()
        else:
            alpha_loss = 0
            alpha = 1

        q_new_actions = torch.min(
            self.qf1(
                obs.unsqueeze(0).repeat(self.ensemble_size, 1, 1),
                new_obs_actions.unsqueeze(0).repeat(self.ensemble_size, 1, 1),
            ),
            self.qf2(
                obs.unsqueeze(0).repeat(self.ensemble_size, 1, 1),
                new_obs_actions.unsqueeze(0).repeat(self.ensemble_size, 1, 1),
            ),
        ).mean(
            0
        )  # (32, 1)

        policy_loss = (alpha * log_pi - q_new_actions).mean()

        """
        QF Loss
        """
        q1_pred = self.qf1(
            obs.unsqueeze(0).repeat(self.ensemble_size, 1, 1),
            actions.unsqueeze(0).repeat(self.ensemble_size, 1, 1),
        )  # (5, 32, 1)
        q2_pred = self.qf2(
            obs.unsqueeze(0).repeat(self.ensemble_size, 1, 1),
            actions.unsqueeze(0).repeat(self.ensemble_size, 1, 1),
        )  # (5, 32, 1)

        next_dist = self.policy(next_obs)
        new_next_actions, new_log_pi = next_dist.rsample_and_logprob()  # (32, 6), (32)
        new_next_actions = new_next_actions.unsqueeze(0).repeat(
            self.ensemble_size, 1, 1
        )
        new_log_pi = new_log_pi.unsqueeze(-1)

        # new_next_actions = torch.stack(new_next_actions_list)
        for_target_q_values = torch.min(
            self.target_qf1(
                next_obs.unsqueeze(0).repeat(self.ensemble_size, 1, 1), new_next_actions
            ),
            self.target_qf2(
                next_obs.unsqueeze(0).repeat(self.ensemble_size, 1, 1), new_next_actions
            ),
        )

        target_q_values = for_target_q_values - alpha * new_log_pi.unsqueeze(0)

        q_target = (
            self.reward_scale * rewards
            + (1.0 - terminals) * self.discount * target_q_values
        )
        qf1_loss = (
            self.qf_criterion(q1_pred, q_target.detach()).sum(0).mean()
        )  # 5, 256, 1
        qf2_loss = (
            self.qf_criterion(q2_pred, q_target.detach()).sum(0).mean()
        )  # 5, 256, 1

        """
        Updating the priority buffer
        """
        with torch.no_grad():
            weight = self.w_activation(self.weight_net(obs, actions))

            normalized_weight = (weight ** (1 / self.temperature)) / (
                (offline_weight ** (1 / self.temperature)).mean() + 1e-10
            )

        new_priority = normalized_weight.clamp(0.001, 1000)

            
        self.priority_replay_buffer.update_priorities(
            batch["tree_idxs"].squeeze().detach().cpu().numpy().astype(np.int),
            new_priority.squeeze().detach().cpu().numpy(),
        )

        """
        Save some statistics for eval
        """
        eval_statistics = OrderedDict()
        if not skip_statistics:
            eval_statistics["QF1 Loss"] = np.mean(ptu.get_numpy(qf1_loss))
            eval_statistics["QF2 Loss"] = np.mean(ptu.get_numpy(qf2_loss))
            eval_statistics["Policy Loss"] = np.mean(ptu.get_numpy(policy_loss))
            eval_statistics["Weight Loss"] = np.mean(ptu.get_numpy(weight_loss))

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
                create_stats_ordered_dict("RL Samples Weights", ptu.get_numpy(weight))
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

            eval_statistics.update(
                create_stats_ordered_dict(
                    "Log Pis",
                    ptu.get_numpy(log_pi),
                )
            )
            policy_statistics = add_prefix(dist.get_diagnostics(), "policy/")
            eval_statistics.update(policy_statistics)
            if self.use_automatic_entropy_tuning:
                eval_statistics["Alpha"] = alpha.item()
                eval_statistics["Alpha Loss"] = alpha_loss.item()

        loss = SACLosses(
            policy_loss=policy_loss,
            qf1_loss=qf1_loss,
            qf2_loss=qf2_loss,
            alpha_loss=alpha_loss,
            weight_loss=weight_loss,
        )

        return loss, eval_statistics

    def get_diagnostics(self):
        stats = super().get_diagnostics()
        stats.update(self.eval_statistics)
        return stats

    def end_epoch(self, epoch):
        self._need_to_update_eval_statistics = True

    def save(self, step):
        pass

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
            self.alpha_optimizer,
            self.qf1_optimizer,
            self.qf2_optimizer,
            self.policy_optimizer,
            self.weight_optimizer,
        ]

    def get_snapshot(self):
        ret = dict(
            qf1=self.qf1,
            qf2=self.qf2,
            target_qf1=self.target_qf1,
            target_qf2=self.target_qf2,
            policy=self.policy,
            weight_net=self.weight_net,
        )
        return ret