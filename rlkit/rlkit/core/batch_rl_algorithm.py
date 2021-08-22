import abc

import numpy as np
import math

import torch
import rlkit.torch.pytorch_util as ptu

import gtimer as gt
from rlkit.core import logger, eval_util
from rlkit.core.rl_algorithm import BaseRLAlgorithm, _get_epoch_timings
from rlkit.data_management.replay_buffer import ReplayBuffer
from rlkit.samplers.data_collector import PathCollector, MdpPathCollector


class BatchRLAlgorithm(BaseRLAlgorithm, metaclass=abc.ABCMeta):
    def __init__(
        self,
        trainer,
        exploration_env,
        evaluation_env,
        exploration_data_collector: PathCollector,
        evaluation_data_collector: PathCollector,
        replay_buffer: ReplayBuffer,
        batch_size,
        max_path_length,
        num_epochs,
        num_eval_steps_per_epoch,
        num_expl_steps_per_train_loop,
        num_trains_per_train_loop,
        num_train_loops_per_epoch=1,
        min_num_steps_before_training=0,
        first_epoch_multiplier=1,
    ):
        super().__init__(
            trainer,
            exploration_env,
            evaluation_env,
            exploration_data_collector,
            evaluation_data_collector,
            replay_buffer,
        )
        self.batch_size = batch_size
        self.max_path_length = max_path_length
        self.num_epochs = num_epochs
        self.num_eval_steps_per_epoch = num_eval_steps_per_epoch
        self.num_trains_per_train_loop = num_trains_per_train_loop
        self.num_train_loops_per_epoch = num_train_loops_per_epoch
        self.num_expl_steps_per_train_loop = num_expl_steps_per_train_loop
        self.min_num_steps_before_training = min_num_steps_before_training

        self.first_epoch_multiplier = first_epoch_multiplier

    def _train(self):
        if self.min_num_steps_before_training > 0:
            init_expl_paths = self.expl_data_collector.collect_new_paths(
                self.max_path_length,
                self.min_num_steps_before_training,
                discard_incomplete_paths=False,
            )
            self.replay_buffer.add_paths(init_expl_paths)
            self.expl_data_collector.end_epoch(-1)

        for epoch in gt.timed_for(
            range(self._start_epoch, self.num_epochs),
            save_itrs=True,
        ):
            self.eval_data_collector.collect_new_paths(
                self.max_path_length,
                self.num_eval_steps_per_epoch,
                discard_incomplete_paths=True,
            )
            gt.stamp("evaluation sampling")

            for _ in range(self.num_train_loops_per_epoch):
                new_expl_paths = self.expl_data_collector.collect_new_paths(
                    self.max_path_length,
                    self.num_expl_steps_per_train_loop,
                    discard_incomplete_paths=False,
                )
                gt.stamp("exploration sampling", unique=False)

                self.replay_buffer.add_paths(new_expl_paths)
                gt.stamp("data storing", unique=False)

                self.training_mode(True)

                if epoch == 0:
                    num_trains_per_train_loop = (
                        self.num_trains_per_train_loop * self.first_epoch_multiplier
                    )
                else:
                    num_trains_per_train_loop = self.num_trains_per_train_loop
                for _ in range(num_trains_per_train_loop):
                    train_data = self.replay_buffer.random_batch(self.batch_size)
                    self.trainer.train(train_data)
                gt.stamp("training", unique=False)
                self.training_mode(False)

            self._end_epoch(epoch)


class OurRLAlgorithm(BaseRLAlgorithm, metaclass=abc.ABCMeta):
    def __init__(
        self,
        trainer,
        exploration_env,
        evaluation_env,
        exploration_data_collector: PathCollector,
        evaluation_data_collector: PathCollector,
        offline_replay_buffer: ReplayBuffer,
        online_replay_buffer: ReplayBuffer,
        priority_replay_buffer: ReplayBuffer,
        batch_size,
        weight_net_batch_size,
        init_online_fraction,
        max_path_length,
        num_epochs,
        num_eval_steps_per_epoch,
        num_expl_steps_per_train_loop,
        num_trains_per_train_loop,
        num_train_loops_per_epoch=1,
        min_num_steps_before_training=0,
        first_epoch_multiplier=1,
    ):
        super().__init__(
            trainer,
            exploration_env,
            evaluation_env,
            exploration_data_collector,
            evaluation_data_collector,
            online_replay_buffer,
        )

        self.offline_replay_buffer = offline_replay_buffer
        self.priority_replay_buffer = priority_replay_buffer
        self.init_online_fraction = init_online_fraction

        """ 
        Important: set transitions.max to be s.t. initial sampling ratio is roughly 0.1
        """
        M = offline_replay_buffer._size
        N = 1000
        # self.priority_replay_buffer.transitions.max = M / (9 * 1000)
        f = init_online_fraction
        self.priority_replay_buffer.transitions.max = M * f / ((1 - f) * N)

        self.batch_size = batch_size
        self.weight_net_batch_size = weight_net_batch_size
        self.max_path_length = max_path_length
        self.num_epochs = num_epochs
        self.num_eval_steps_per_epoch = num_eval_steps_per_epoch
        self.num_trains_per_train_loop = num_trains_per_train_loop
        self.num_train_loops_per_epoch = num_train_loops_per_epoch
        self.num_expl_steps_per_train_loop = num_expl_steps_per_train_loop
        self.min_num_steps_before_training = min_num_steps_before_training
        self.first_epoch_multiplier = first_epoch_multiplier

    def _train(self):

        if self.min_num_steps_before_training > 0:
            init_expl_paths = self.expl_data_collector.collect_new_paths(
                self.max_path_length,
                self.min_num_steps_before_training,
                discard_incomplete_paths=False,
            )
            self.replay_buffer.add_paths(init_expl_paths)
            self.priority_replay_buffer.add_paths(init_expl_paths)
            self.expl_data_collector.end_epoch(-1)

        for epoch in gt.timed_for(
            range(self._start_epoch, self.num_epochs),
            save_itrs=True,
        ):
            self.eval_data_collector.collect_new_paths(
                self.max_path_length,
                self.num_eval_steps_per_epoch,
                discard_incomplete_paths=True,
            )
            gt.stamp("evaluation sampling")

            for _ in range(self.num_train_loops_per_epoch):
                new_expl_paths = self.expl_data_collector.collect_new_paths(
                    self.max_path_length,
                    self.num_expl_steps_per_train_loop,
                    discard_incomplete_paths=False,
                )
                gt.stamp("exploration sampling", unique=False)

                self.replay_buffer.add_paths(new_expl_paths)
                self.priority_replay_buffer.add_paths(new_expl_paths)
                gt.stamp("data storing", unique=False)

                self.training_mode(True)
                if epoch == 0:
                    num_trains_per_train_loop = (
                        self.num_trains_per_train_loop * self.first_epoch_multiplier
                    )
                else:
                    num_trains_per_train_loop = self.num_trains_per_train_loop
                for _ in range(num_trains_per_train_loop):
                    train_data_online = self.replay_buffer.random_batch(
                        self.weight_net_batch_size
                    )
                    train_data_offline = self.offline_replay_buffer.random_batch(
                        self.weight_net_batch_size
                    )
                    train_data_rl = self.priority_replay_buffer.random_batch(
                        self.batch_size
                    )

                    train_data = dict()
                    train_data["offline_observations"] = train_data_offline[
                        "observations"
                    ]
                    train_data["offline_next_observations"] = train_data_offline[
                        "next_observations"
                    ]
                    train_data["offline_actions"] = train_data_offline["actions"]
                    train_data["offline_rewards"] = train_data_offline["rewards"]
                    train_data["offline_terminals"] = train_data_offline["terminals"]
                    train_data["online_observations"] = train_data_online[
                        "observations"
                    ]
                    train_data["online_next_observations"] = train_data_online[
                        "next_observations"
                    ]
                    train_data["online_actions"] = train_data_online["actions"]
                    train_data["online_rewards"] = train_data_online["rewards"]
                    train_data["online_terminals"] = train_data_online["terminals"]

                    train_data["rl_observations"] = train_data_rl["observations"]
                    train_data["rl_next_observations"] = train_data_rl[
                        "next_observations"
                    ]
                    train_data["rl_actions"] = train_data_rl["actions"]
                    train_data["rl_rewards"] = train_data_rl["rewards"]
                    train_data["rl_terminals"] = train_data_rl["terminals"]
                    train_data["idxs"] = train_data_rl["idxs"]
                    train_data["tree_idxs"] = train_data_rl["tree_idxs"]

                    self.trainer.train(train_data)

                gt.stamp("training", unique=False)
                self.training_mode(False)

            self._end_epoch(epoch)
