import sys
import math
import random
import numpy as np

import sys
import math

from rlkit.data_management.replay_buffer import ReplayBuffer


def list_to_dict(in_list):
    return dict((i, in_list[i]) for i in range(0, len(in_list)))


def exchange_key_value(in_dict):
    return dict((in_dict[i], i) for i in in_dict)


class BinaryHeap(object):
    def __init__(self, priority_size=100, replace=True):
        self.e2p = {}
        self.p2e = {}
        self.replace = replace

        self.priority_queue = {}  # key = data, value = index
        self.size = 0
        self.max_size = priority_size

    def __repr__(self):
        """
        :return: string of the priority queue, with level info
        """
        if self.size == 0:
            return "No element in heap!"
        to_string = ""
        level = -1
        max_level = int(math.floor(math.log(self.size, 2)))

        for i in range(1, self.size + 1):
            now_level = int(math.floor(math.log(i, 2)))
            if level != now_level:
                to_string = (
                    to_string
                    + ("\n" if level != -1 else "")
                    + "    " * (max_level - now_level)
                )
                level = now_level

            to_string = (
                to_string
                + "%.2f " % self.priority_queue[i][1]
                + "    " * (max_level - now_level)
            )

        return to_string

    def check_full(self):
        return self.size > self.max_size

    def _insert(self, priority, e_id):
        """
        insert new experience id with priority
        (maybe don't need get_max_priority and implement it in this function)
        :param priority: priority value
        :param e_id: experience id
        :return: bool
        """
        self.size += 1

        if self.check_full() and not self.replace:
            sys.stderr.write(
                "Error: no space left to add experience id %d with priority value %f\n"
                % (e_id, priority)
            )
            return False
        else:
            self.size = min(self.size, self.max_size)

        self.priority_queue[self.size] = (priority, e_id)
        self.p2e[self.size] = e_id
        self.e2p[e_id] = self.size

        self.up_heap(self.size)
        return True

    def update(self, priority, e_id):
        """
        update priority value according its experience id
        :param priority: new priority value
        :param e_id: experience id
        :return: bool
        """
        if e_id in self.e2p:
            p_id = self.e2p[e_id]
            self.priority_queue[p_id] = (priority, e_id)
            self.p2e[p_id] = e_id

            self.down_heap(p_id)
            self.up_heap(p_id)
            return True
        else:
            # this e id is new, do insert
            return self._insert(priority, e_id)

    def get_max_priority(self):
        """
        get max priority, if no experience, return 1
        :return: max priority if size > 0 else 1
        """
        if self.size > 0:
            return self.priority_queue[1][0]
        else:
            return 1

    def pop(self):
        """
        pop out the max priority value with its experience id
        :return: priority value & experience id
        """
        if self.size == 0:
            sys.stderr.write("Error: no value in heap, pop failed\n")
            return False, False

        pop_priority, pop_e_id = self.priority_queue[1]
        self.e2p[pop_e_id] = -1
        # replace first
        last_priority, last_e_id = self.priority_queue[self.size]
        self.priority_queue[1] = (last_priority, last_e_id)
        self.size -= 1
        self.e2p[last_e_id] = 1
        self.p2e[1] = last_e_id

        self.down_heap(1)

        return pop_priority, pop_e_id

    def up_heap(self, i):
        """
        upward balance
        :param i: tree node i
        :return: None
        """
        if i > 1:
            parent = math.floor(i / 2)
            if self.priority_queue[parent][0] < self.priority_queue[i][0]:
                tmp = self.priority_queue[i]
                self.priority_queue[i] = self.priority_queue[parent]
                self.priority_queue[parent] = tmp
                # change e2p & p2e
                self.e2p[self.priority_queue[i][1]] = i
                self.e2p[self.priority_queue[parent][1]] = parent
                self.p2e[i] = self.priority_queue[i][1]
                self.p2e[parent] = self.priority_queue[parent][1]
                # up heap parent
                self.up_heap(parent)

    def down_heap(self, i):
        """
        downward balance
        :param i: tree node i
        :return: None
        """
        if i < self.size:
            greatest = i
            left, right = i * 2, i * 2 + 1
            if (
                left < self.size
                and self.priority_queue[left][0] > self.priority_queue[greatest][0]
            ):
                greatest = left
            if (
                right < self.size
                and self.priority_queue[right][0] > self.priority_queue[greatest][0]
            ):
                greatest = right

            if greatest != i:
                tmp = self.priority_queue[i]
                self.priority_queue[i] = self.priority_queue[greatest]
                self.priority_queue[greatest] = tmp
                # change e2p & p2e
                self.e2p[self.priority_queue[i][1]] = i
                self.e2p[self.priority_queue[greatest][1]] = greatest
                self.p2e[i] = self.priority_queue[i][1]
                self.p2e[greatest] = self.priority_queue[greatest][1]
                # down heap greatest
                self.down_heap(greatest)

    def get_priority(self):
        """
        get all priority value
        :return: list of priority
        """
        return list(map(lambda x: x[0], self.priority_queue.values()))[0 : self.size]

    def get_e_id(self):
        """
        get all experience id in priority queue
        :return: list of experience ids order by their priority
        """
        return list(map(lambda x: x[1], self.priority_queue.values()))[0 : self.size]

    def balance_tree(self):
        """
        rebalance priority queue
        :return: None
        """
        sort_array = sorted(
            self.priority_queue.values(), key=lambda x: x[0], reverse=True
        )
        # reconstruct priority_queue
        self.priority_queue.clear()
        self.p2e.clear()
        self.e2p.clear()
        cnt = 1
        while cnt <= self.size:
            priority, e_id = sort_array[cnt - 1]
            self.priority_queue[cnt] = (priority, e_id)
            self.p2e[cnt] = e_id
            self.e2p[e_id] = cnt
            cnt += 1
        # sort the heap
        for i in range(int(math.floor(self.size / 2)), 1, -1):
            self.down_heap(i)

    def priority_to_experience(self, priority_ids):
        """
        retrieve experience ids by priority ids
        :param priority_ids: list of priority id
        :return: list of experience id
        """
        return [self.p2e[i] for i in priority_ids]


class BinaryHeapReplayBuffer(ReplayBuffer):
    def __init__(self, max_replay_buffer_size, batch_size, env):

        self.size = self.priority_size = max_replay_buffer_size
        self.replace_flag = True

        self.env = env
        self._ob_space = env.observation_space
        self._action_space = env.action_space

        self.alpha = 0.7  # ?
        # self.beta_zero = 0.5  # ?
        self.batch_size = batch_size
        self.learn_start = 0
        # self.total_steps = 100000
        # partition number N, split total size to N part
        self.partition_num = batch_size  # as suggested by PER

        self.index = 0
        self.record_size = 0
        self.isFull = False

        self._experience = {}
        self.priority_queue = BinaryHeap(self.priority_size)
        self.distributions = self.build_distributions()

        # self.beta_grad = (1 - self.beta_zero) / float(
        #     self.total_steps - self.learn_start
        # )

    def build_distributions(self):
        """
        preprocess pow of rank
        (rank i) ^ (-alpha) / sum ((rank i) ^ (-alpha))
        :return: distributions, dict
        """
        res = {}
        n_partitions = self.partition_num
        partition_num = 1
        # each part size
        partition_size = int(math.floor(self.size / n_partitions))

        for n in range(partition_size, self.size + 1, partition_size):
            if self.learn_start <= n <= self.priority_size:
                distribution = {}
                # P(i) = (rank i) ^ (-alpha) / sum ((rank i) ^ (-alpha))
                pdf = list(map(lambda x: math.pow(x, -self.alpha), range(1, n + 1)))
                pdf_sum = math.fsum(pdf)
                distribution["pdf"] = list(map(lambda x: x / pdf_sum, pdf))
                # split to k segment, and than uniform sample in each k
                # set k = batch_size, each segment has total probability is 1 / batch_size
                # strata_ends keep each segment start pos and end pos
                cdf = np.cumsum(distribution["pdf"])
                strata_ends = {1: 0, self.batch_size + 1: n}
                step = 1 / float(self.batch_size)
                index = 1
                for s in range(2, self.batch_size + 1):
                    while cdf[index] < step:
                        index += 1
                    strata_ends[s] = index
                    step += 1 / float(self.batch_size)

                distribution["strata_ends"] = strata_ends

                res[partition_num] = distribution

            partition_num += 1

        return res

    def fix_index(self):
        """
        get next insert index
        :return: index, int
        """
        if self.record_size <= self.size:
            self.record_size += 1
        if self.index % self.size == 0:
            self.isFull = True if len(self._experience) == self.size else False
            if self.replace_flag:
                self.index = 1
                return self.index
            else:
                sys.stderr.write(
                    "Experience replay buff is full and replace is set to FALSE!\n"
                )
                return -1
        else:
            self.index += 1
            return self.index

    def add_sample(
        self, observation, action, reward, terminal, next_observation, **kwargs
    ):
        """
        store experience, suggest that experience is a tuple of (s1, a, r, s2, t)
        so each experience is valid
        :param experience: maybe a tuple, or list
        :return: bool, indicate insert status
        """
        experience = (observation, action, reward, terminal, next_observation)
        insert_index = self.fix_index()
        if insert_index > 0:
            if insert_index in self._experience:
                del self._experience[insert_index]
            self._experience[insert_index] = experience
            # add to priority queue
            priority = self.priority_queue.get_max_priority()
            self.priority_queue.update(priority, insert_index)
            return True
        else:
            sys.stderr.write("Insert failed\n")
            return False

    def retrieve(self, indices):
        """
        get experience from indices
        :param indices: list of experience id
        :return: experience replay sample
        """
        return [self._experience[v] for v in indices]

    def rebalance(self):
        """
        rebalance priority queue
        :return: None
        """
        self.priority_queue.balance_tree()

    def update_priorities(self, indices, values):
        """
        update priority according indices and deltas
        :param indices: list of experience id
        :param delta: list of delta, order correspond to indices
        :return: None
        """
        for i in range(0, len(indices)):
            self.priority_queue.update(values[i], indices[i])

    def random_batch(self, batch_size=None):
        # batch_size is just a dummy variable here
        """
        sample a mini batch from experience replay
        :param global_step: now training step
        :return: experience, list, samples
        :return: w, list, weights
        :return: rank_e_id, list, samples id, used for update priority
        """
        if self.record_size < self.learn_start:
            sys.stderr.write("Record size less than learn start! Sample failed\n")
            return False, False, False

        dist_index = math.floor(self.record_size / self.size * self.partition_num)
        # issue 1 by @camigord
        partition_size = math.floor(self.size / self.partition_num)
        partition_max = dist_index * partition_size
        distribution = self.distributions[dist_index]
        rank_list = []
        # sample from k segments - needs debug
        for n in range(1, self.batch_size + 1):
            index = random.randint(
                distribution["strata_ends"][n] + 1, distribution["strata_ends"][n + 1]
            )
            rank_list.append(index)

        # beta, increase by global_step, max 1
        # beta = min(
        #     self.beta_zero + (global_step - self.learn_start - 1) * self.beta_grad, 1
        # )
        # beta = 1.0
        # find all alpha pow, notice that pdf is a list, start from 0
        alpha_pow = [distribution["pdf"][v - 1] for v in rank_list]
        # w = (N * P(i)) ^ (-beta) / max w
        # w = np.power(np.array(alpha_pow) * partition_max, -beta)
        # w_max = max(w)
        # w = np.divide(w, w_max)
        # rank list is priority id
        # convert to experience id
        rank_e_id = self.priority_queue.priority_to_experience(rank_list)

        # get experience id according rank_e_id

        experience = self.retrieve(rank_e_id)
        experience = list(zip(*experience))

        batch = dict()
        batch["observations"] = np.array(experience[0])
        batch["actions"] = np.array(experience[1])
        batch["rewards"] = np.array(experience[2]).reshape(-1, 1)
        batch["terminals"] = np.array(experience[3]).reshape(-1, 1)
        batch["next_observations"] = np.array(experience[4])
        batch["tree_idxs"] = batch["idxs"] = np.array(rank_e_id)

        # (observation, action, reward, terminal, next_observation)
        # return experience, w, rank_e_id
        return batch

    def terminate_episode(self):
        pass

    def num_steps_can_sample(self):
        return len(self._experience)

    def get_diagnostics(self):
        return OrderedDict([("size", len(self._experience))])


import gym, d4rl

if __name__ == "__main__":
    env = gym.make("walker2d-medium-replay-v0")
    replay_buffer = BinaryHeapReplayBuffer(50000, 4, env)
    ds = env.get_dataset()
    dataset_size = ds["observations"].shape[0]
    for i in range(dataset_size - 1):
        obs = ds["observations"][i]
        new_obs = ds["observations"][i + 1]
        action = ds["actions"][i]
        reward = ds["rewards"][i]
        done = ds["terminals"][i]
        timeout = ds["timeouts"][i]
        if timeout:
            continue
        replay_buffer.add_sample(obs, action, reward, done, new_obs)

    print(replay_buffer.random_batch()["observations"])
    print(replay_buffer.random_batch()["actions"])
    print(replay_buffer.random_batch()["rewards"])
    print(replay_buffer.random_batch()["terminals"])
    print(replay_buffer.random_batch()["idxs"])
