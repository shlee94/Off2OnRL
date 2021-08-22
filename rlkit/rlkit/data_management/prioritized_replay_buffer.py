"""
Adapted from: https://github.com/Kaixhin/Rainbow/blob/master/memory.py
"""

from gym.spaces import Discrete

from rlkit.data_management.replay_buffer import ReplayBuffer

from rlkit.envs.env_utils import get_dim
import numpy as np


class SegmentTree:
    def __init__(
        self, max_size, observation_dim, action_dim
    ):
        self._observations = np.zeros((max_size, observation_dim))
        self._next_obs = np.zeros((max_size, observation_dim))
        self._actions = np.zeros((max_size, action_dim))
        self._rewards = np.zeros((max_size, 1))
        self._terminals = np.zeros((max_size, 1), dtype="uint8")

        self.index = 0
        self.max_size = max_size
        self.full = False
        self.tree_start = 2 ** (max_size - 1).bit_length() - 1
        self.sum_tree = np.zeros((self.tree_start + self.max_size), dtype=np.float32)

        self.max = 1.0

    def _update_nodes(self, indices):
        children_indices = indices * 2 + np.expand_dims([1, 2], axis=1)
        # [0,1,2,3] -> [1,3,5,7; 2,4,6,8]
        self.sum_tree[indices] = np.sum(self.sum_tree[children_indices], axis=0)

    def _propagate(self, indices):
        parents = (indices - 1) // 2
        unique_parents = np.unique(parents)
        self._update_nodes(unique_parents)
        if parents[0] != 0:
            self._propagate(parents)

    def _propagate_index(self, index):
        parent = (index - 1) // 2
        left, right = 2 * parent + 1, 2 * parent + 2
        self.sum_tree[parent] = self.sum_tree[left] + self.sum_tree[right]
        if parent != 0:
            self._propagate_index(parent)

    def update(self, indices, values):
        self.sum_tree[indices] = values
        self._propagate(indices)
        current_max_value = np.max(values)
        self.max = max(current_max_value, self.max)

    # update single value given a tree index for efficiency
    def _update_index(self, index, value):
        self.sum_tree[index] = value  # set new value
        self._propagate_index(index)  # propagate value
        self.max = max(value, self.max)

    def append(
        self,
        observation,
        action,
        reward,
        terminal,
        next_obs,
        value,
        **kwargs
    ):
        self._observations[self.index] = observation
        self._actions[self.index] = action
        self._rewards[self.index] = reward
        self._terminals[self.index] = terminal
        self._next_obs[self.index] = next_obs

        self._update_index(self.index + self.tree_start, value)
        self.index = (self.index + 1) % self.max_size
        self.full = self.full or self.index == 0
        self.max = max(value, self.max)

    def _retrieve(self, indices, values):
        children_indices = indices * 2 + np.expand_dims(
            [1, 2], axis=1
        )  # Make matrix of children indices
        if children_indices[0, 0] >= self.sum_tree.shape[0]:
            return indices
        left_children_values = self.sum_tree[children_indices[0]]
        successor_choices = np.greater(values, left_children_values).astype(
            np.int32
        )  # Classify which values are in left or right branches
        successor_indices = children_indices[
            successor_choices, np.arange(indices.size)
        ]  # Use classification to index into the indices matrix
        successor_values = (
            values - successor_choices * left_children_values
        )  # Subtract the left branch values when searching in the right branch
        return self._retrieve(successor_indices, successor_values)

    def find(self, values):
        indices = self._retrieve(np.zeros(values.shape, dtype=np.int32), values)
        data_index = indices - self.tree_start
        return (self.sum_tree[indices], data_index, indices)

    def get(self, data_index):
        batch = dict()
        batch["observations"] = self._observations[data_index]
        batch["next_observations"] = self._next_obs[data_index]
        batch["actions"] = self._actions[data_index]
        batch["rewards"] = self._rewards[data_index]
        batch["terminals"] = self._terminals[data_index]
        return batch

    def total(self):
        return self.sum_tree[0]


class PriorityReplayBuffer(ReplayBuffer):
    def __init__(self, max_replay_buffer_size, env, env_info_sizes=None):
        """
        :param max_replay_buffer_size:
        :param env:
        """
        self.env = env
        self._ob_space = env.observation_space
        self._action_space = env.action_space

        self.transitions = SegmentTree(
            max_replay_buffer_size, get_dim(self._ob_space), get_dim(self._action_space)
        )

    def add_sample(
        self, observation, action, reward, terminal, next_observation, **kwargs
    ):
        if isinstance(self._action_space, Discrete):
            new_action = np.zeros(self._action_dim)
            new_action[action] = 1
        else:
            new_action = action
        self.transitions.append(
            observation,
            action,
            reward,
            terminal,
            next_observation,
            self.transitions.max,
        )

    def _get_transitions(self, idxs):
        transitions = self.transitions.get(data_index=idxs)
        return transitions

    def _get_samples_from_segments(self, batch_size, p_total):
        segment_length = p_total / batch_size
        segment_starts = np.arange(batch_size) * segment_length
        valid = False
        while not valid:
            samples = (
                np.random.uniform(0.0, segment_length, [batch_size]) + segment_starts
            )
            probs, idxs, tree_idxs = self.transitions.find(samples)
            if np.all(probs != 0):
                valid = True
        batch = self._get_transitions(idxs)
        batch["idxs"] = idxs
        batch["tree_idxs"] = tree_idxs

        return batch

    def random_batch(self, batch_size):
        # return tree_idxs s.t. their values can be updated
        p_total = self.transitions.total()
        return self._get_samples_from_segments(batch_size, p_total)

    def update_priorities(self, idxs, priorities):
        self.transitions.update(idxs, priorities)

    def terminate_episode(self):
        pass

    def num_steps_can_sample(self):
        return self.transitions.index

    def get_diagnostics(self):
        return OrderedDict([("size", self.transitions.index)])

