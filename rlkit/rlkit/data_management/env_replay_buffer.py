from gym.spaces import Discrete

from rlkit.data_management.simple_replay_buffer import SimpleReplayBuffer
from rlkit.envs.env_utils import get_dim
import numpy as np


class EnvReplayBuffer(SimpleReplayBuffer):
    def __init__(self, max_replay_buffer_size, env, env_info_sizes=None):
        """
        :param max_replay_buffer_size:
        :param env:
        """
        self.env = env
        self._ob_space = env.observation_space
        self._action_space = env.action_space

        if env_info_sizes is None:
            if hasattr(env, "info_sizes"):
                env_info_sizes = env.info_sizes
            else:
                env_info_sizes = dict()

        super().__init__(
            max_replay_buffer_size=max_replay_buffer_size,
            observation_dim=get_dim(self._ob_space),
            action_dim=get_dim(self._action_space),
            env_info_sizes=env_info_sizes,
        )

    def add_sample(
        self, observation, action, reward, terminal, next_observation, **kwargs
    ):
        if isinstance(self._action_space, Discrete):
            new_action = np.zeros(self._action_dim)
            new_action[action] = 1
        else:
            new_action = action
        return super().add_sample(
            observation=observation,
            action=new_action,
            reward=reward,
            next_observation=next_observation,
            terminal=terminal,
            **kwargs
        )


class EnvMaskedReplayBuffer(SimpleReplayBuffer):
    def __init__(
        self,
        max_replay_buffer_size,
        env,
        ensemble_size,
        masking_probability,
        env_info_sizes=None,
    ):
        """
        :param max_replay_buffer_size:
        """
        self.env = env
        self.masking_probability = masking_probability
        self.ensemble_size = ensemble_size
        self._ob_space = env.observation_space
        self._action_space = env.action_space

        if env_info_sizes is None:
            if hasattr(env, "info_sizes"):
                env_info_sizes = env.info_sizes
            else:
                env_info_sizes = dict()

        super().__init__(
            max_replay_buffer_size=max_replay_buffer_size,
            observation_dim=get_dim(self._ob_space),
            action_dim=get_dim(self._action_space),
            env_info_sizes=env_info_sizes,
        )
        self._masks = np.zeros((max_replay_buffer_size, ensemble_size))

    def add_sample(
        self,
        observation,
        action,
        reward,
        terminal,
        next_observation,
        mask,
        env_info=None,
        **kwargs
    ):

        if isinstance(self._action_space, Discrete):
            new_action = np.zeros(self._action_dim)
            new_action[action] = 1
        else:
            new_action = action

        self._observations[self._top] = observation
        self._actions[self._top] = new_action
        self._rewards[self._top] = reward
        self._terminals[self._top] = terminal
        self._next_obs[self._top] = next_observation
        self._masks[self._top] = mask

        for key in self._env_info_keys:
            self._env_infos[key][self._top] = env_info[key]
        super()._advance()

    def random_batch(self, batch_size):
        indices = np.random.choice(
            self._size,
            size=batch_size,
            replace=self._replace or self._size < batch_size,
        )
        if not self._replace and self._size < batch_size:
            warnings.warn(
                "Replace was set to false, but is temporarily set to true because batch size is larger than current size of replay."
            )
        batch = dict(
            observations=self._observations[indices],
            actions=self._actions[indices],
            rewards=self._rewards[indices],
            terminals=self._terminals[indices],
            next_observations=self._next_obs[indices],
            masks=self._masks[indices],
        )
        for key in self._env_info_keys:
            assert key not in batch.keys()
            batch[key] = self._env_infos[key][indices]
        return batch

    def add_path(self, path):
        for i, (
            obs,
            action,
            reward,
            next_obs,
            terminal,
            agent_info,
            env_info,
        ) in enumerate(
            zip(
                path["observations"],
                path["actions"],
                path["rewards"],
                path["next_observations"],
                path["terminals"],
                path["agent_infos"],
                path["env_infos"],
            )
        ):
            mask = np.random.binomial(
                1, self.masking_probability, size=self.ensemble_size
            )
            if mask.sum() == 0:
                mask = np.zeros(self.ensemble_size)
                mask[np.random.randint(self.ensemble_size)] = 1
            self.add_sample(
                observation=obs,
                action=action,
                reward=reward,
                terminal=terminal,
                next_observation=next_obs,
                mask=np.random.binomial(
                    1, self.masking_probability, size=self.ensemble_size
                ),
                agent_info=agent_info,
                env_info=env_info,
            )
        self.terminate_episode()