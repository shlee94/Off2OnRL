import gym
# import roboverse
# from rlkit.data_management.awr_env_replay_buffer import AWREnvReplayBuffer
from rlkit.data_management.env_replay_buffer import EnvReplayBuffer
from rlkit.data_management.split_buffer import SplitReplayBuffer
from rlkit.envs.wrappers import NormalizedBoxEnv, StackObservationEnv, RewardWrapperEnv
import rlkit.torch.pytorch_util as ptu
from rlkit.samplers.data_collector import MdpPathCollector, ObsDictPathCollector
from rlkit.samplers.data_collector.step_collector import MdpStepCollector
from rlkit.torch.networks import ConcatMlp
from rlkit.torch.sac.policies import TanhGaussianPolicy, MakeDeterministic
from rlkit.torch.sac.awac_ensemble_trainer import AWACEnsembleTrainer
from rlkit.torch.torch_rl_algorithm import (
    TorchBatchRLAlgorithm,
    TorchOnlineRLAlgorithm,
    TorchSeparateBuffersRLAlgorithm,
)

from rlkit.torch.core import torch_ify, elem_or_tuple_to_numpy

from pathlib import Path

from rlkit.torch.distributions import (
    Delta, TanhNormal, MultivariateDiagonalNormal, GaussianMixture, GaussianMixtureFull,
)

from rlkit.demos.source.hdf5_path_loader import HDF5PathLoader
from rlkit.demos.source.mdp_path_loader import MDPPathLoader
# from rlkit.visualization.video import save_paths, VideoSaveFunction

# from multiworld.core.image_env import ImageEnv

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torchvision.utils import save_image

from rlkit.exploration_strategies.base import \
    PolicyWrappedWithExplorationStrategy
from rlkit.exploration_strategies.gaussian_and_epsilon_strategy import GaussianAndEpsilonStrategy
from rlkit.exploration_strategies.ou_strategy import OUStrategy

import os.path as osp
from rlkit.core import logger
from rlkit.util.io import load_local_or_remote_file
import pickle

# from rlkit.envs.images import Renderer, InsertImageEnv, EnvRenderer
from rlkit.envs.make_env import make


class DenseLayer(nn.Module):
    def __init__(self, ensemble_size, input_dim, output_dim, activation=F.relu, hidden_init=ptu.fanin_init, bias_const=0.):
        super(DenseLayer, self).__init__()
        self.ensemble_size = ensemble_size
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.activation = activation

        weights = []
        biases = []
        for _ in range(ensemble_size):
            weight = torch.zeros(input_dim, output_dim)
            hidden_init(weight)
            weights.append(weight.unsqueeze(0))
            bias = torch.zeros(1, output_dim).unsqueeze(0) + bias_const
            biases.append(bias)
        self.weights = torch.cat(weights) # num_ensemble_size, input_dim, output_dim
        self.weights = torch.nn.Parameter(data=self.weights, requires_grad=True)
        self.biases = torch.cat(biases) # num_ensemble_size, 1, output_dim
        self.biases = torch.nn.Parameter(data=self.biases, requires_grad=True)

    def forward(self, x):
        # x: (num_ensemble_size, batch_size, indim), weights: (num_ensemble_size, indim, outdim), biases: (num_ensemble_size, 1, outdim)
        # (5, 36, 17), (5, 17, 10) -> (5, 36, 10)
        x = torch.einsum('bij,bjk->bik', [x, self.weights]) + self.biases
        if self.activation is None:
            return x
        else:
            return self.activation(x) # (num_ensemble_size, batch_size, out_dim)


class ParallelMlp(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, ensemble_size, bias_const=0., init_w=3e-3):
        super(ParallelMlp, self).__init__()

        self.ensemble_size = ensemble_size
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.init_w = init_w

        self.h1 = DenseLayer(ensemble_size, input_dim=input_dim, output_dim=hidden_dim, hidden_init=ptu.fanin_init)
        self.h2 = DenseLayer(ensemble_size, input_dim=hidden_dim, output_dim=hidden_dim, hidden_init=ptu.fanin_init)
        self.output = DenseLayer(ensemble_size, input_dim=hidden_dim, output_dim=output_dim, activation=None, hidden_init=self.last_fc_init, bias_const=bias_const)

    def last_fc_init(self, tensor):
        tensor.data.uniform_(-self.init_w, self.init_w)

    def forward(self, x1, x2):
        x = torch.cat([x1, x2], dim=-1)
        x = self.h1(x)
        x = self.h2(x)
        x = self.output(x)
        return x
    
    def weight_norm(self):
        return torch.norm(self.h1.weights) + torch.norm(self.h2.weights) + torch.norm(self.output.weights)



class ParallelGaussianPolicy:
    """
    This is only used for MDPCollectors
    NOTE(Younggyo): I'm not sure averaging standard deviation of independent gaussian policies is correct
    Not sure this will work
    """
    def __init__(
        self,
        policies
    ):
        self.policies = policies
    
    def forward(self, obs):
        mean_list, std_list = [], []
        for policy in self.policies:
            multdiagnormal = policy(obs)
            mean = multdiagnormal.mean
            std = multdiagnormal.distribution.stddev
            mean_list.append(mean)
            std_list.append(std)
        
        means = torch.cat(mean_list)
        stds = torch.cat(std_list)

        avg_mean = means.mean(0).unsqueeze(0)
        avg_var = (means ** 2 + stds ** 2).mean(0).unsqueeze(0) - avg_mean ** 2
        avg_std = avg_var.sqrt()

        return MultivariateDiagonalNormal(avg_mean, avg_std)
    

    def logprob(self, action, mean, std):
        normal = MultivariateDiagonalNormal(mean, std)
        log_prob = normal.log_prob(
            action,
        )
        log_prob = log_prob.sum(dim=1, keepdim=True)
        return log_prob
    
    def get_action(self, obs_np, ):
        actions = self.get_actions(obs_np[None])
        return actions[0, :], {}

    def get_actions(self, obs_np, ):
        dist = self._get_dist_from_np(obs_np)
        actions = dist.sample()
        return elem_or_tuple_to_numpy(actions)

    def _get_dist_from_np(self, *args, **kwargs):
        torch_args = tuple(torch_ify(x) for x in args)
        torch_kwargs = {k: torch_ify(v) for k, v in kwargs.items()}
        dist = self.forward(*torch_args, **torch_kwargs)
        return dist
    
    def reset(self):
        pass




















ENV_PARAMS = {
    'HalfCheetah-v2': {
        'num_expl_steps_per_train_loop': 1000,
        'max_path_length': 1000,
        'env_demo_path': dict(
            path="demos/icml2020/mujoco/hc_action_noise_15.npy",
            # path='~/.d4rl/datasets/halfcheetah_random.hdf5',
            obs_dict=False,
            is_demo=True,
        ),
        'env_offpolicy_data_path': dict(
            path="demos/icml2020/mujoco/hc_off_policy_15_demos_100.npy",
            obs_dict=False,
            is_demo=False,
            train_split=0.9,
        ),
    },
    'Ant-v2': {
        'num_expl_steps_per_train_loop': 1000,
        'max_path_length': 1000,
        'env_demo_path': dict(
            path="demos/icml2020/mujoco/ant_action_noise_15.npy",
            obs_dict=False,
            is_demo=True,
        ),
        'env_offpolicy_data_path': dict(
            path="demos/icml2020/mujoco/ant_off_policy_15_demos_100.npy",
            obs_dict=False,
            is_demo=False,
            train_split=0.9,
        ),
    },
    'Walker2d-v2': {
        'num_expl_steps_per_train_loop': 1000,
        'max_path_length': 1000,
        'env_demo_path': dict(
            path="demos/icml2020/mujoco/walker_action_noise_15.npy",
            obs_dict=False,
            is_demo=True,
        ),
        'env_offpolicy_data_path': dict(
            path="demos/icml2020/mujoco/walker_off_policy_15_demos_100.npy",
            obs_dict=False,
            is_demo=False,
            train_split=0.9,
        ),
    },

    'SawyerRigGrasp-v0': {
        'env_id': 'SawyerRigGrasp-v0',
        # 'num_expl_steps_per_train_loop': 1000,
        'max_path_length': 50,
        # 'num_epochs': 1000,
    },

    'pen-binary-v0': {
        'env_id': 'pen-binary-v0',
        'max_path_length': 200,
        'env_demo_path': dict(
            path="demos/icml2020/hand/pen2_sparse.npy",
            obs_dict=True,
            is_demo=True,
        ),
        'env_offpolicy_data_path': dict(
            path="demos/icml2020/hand/pen_bc_sparse4.npy",
            obs_dict=False,
            is_demo=False,
            train_split=0.9,
        ),
    },
    'door-binary-v0': {
        'env_id': 'door-binary-v0',
        'max_path_length': 200,
        'env_demo_path': dict(
            path="demos/icml2020/hand/door2_sparse.npy",
            obs_dict=True,
            is_demo=True,
        ),
        'env_offpolicy_data_path': dict(
            path="demos/icml2020/hand/door_bc_sparse4.npy",
            obs_dict=False,
            is_demo=False,
            train_split=0.9,
        ),
    },
    'relocate-binary-v0': {
        'env_id': 'relocate-binary-v0',
        'max_path_length': 200,
        'env_demo_path': dict(
            path="demos/icml2020/hand/relocate2_sparse.npy",
            obs_dict=True,
            is_demo=True,
        ),
        'env_offpolicy_data_path': dict(
            path="demos/icml2020/hand/relocate_bc_sparse4.npy",
            obs_dict=False,
            is_demo=False,
            train_split=0.9,
        ),
    },
}


def process_args(variant):
    if variant.get("debug", False):
        variant['max_path_length'] = 50
        variant['batch_size'] = 5
        variant['num_epochs'] = 5
        # variant['num_eval_steps_per_epoch'] = 100
        # variant['num_expl_steps_per_train_loop'] = 100
        variant['num_trains_per_train_loop'] = 10
        # variant['min_num_steps_before_training'] = 100
        variant['trainer_kwargs']['bc_num_pretrain_steps'] = min(10, variant['trainer_kwargs'].get('bc_num_pretrain_steps', 0))
        variant['trainer_kwargs']['q_num_pretrain1_steps'] = min(10, variant['trainer_kwargs'].get('q_num_pretrain1_steps', 0))
        variant['trainer_kwargs']['q_num_pretrain2_steps'] = min(10, variant['trainer_kwargs'].get('q_num_pretrain2_steps', 0))

def resume(variant):
    data = load_local_or_remote_file(variant.get("pretrained_algorithm_path"), map_location="cuda")
    algo = data['algorithm']

    algo.num_epochs = variant['num_epochs']

    post_pretrain_hyperparams = variant["trainer_kwargs"].get("post_pretrain_hyperparams", {})
    algo.trainer.set_algorithm_weights(**post_pretrain_hyperparams)

    algo.train()







def experiment(variant):
    if variant.get("pretrained_algorithm_path", False):
        resume(variant)
        return

    normalize_env = variant.get('normalize_env', True)
    seed = int(variant.get('seed', 0))
    env_id = variant.get('env_id', None)
    # env_params = ENV_PARAMS.get(env_id, {})
    # variant.update(env_params)
    env_class = variant.get('env_class', None)
    env_kwargs = variant.get('env_kwargs', {})

    expl_env = make(env_id, env_class, env_kwargs, normalize_env)
    expl_env.seed(seed)
    eval_env = make(env_id, env_class, env_kwargs, normalize_env)
    eval_env.seed(seed)





    # For loading demo paths
    if variant.get('add_env_demos', False):
        variant["path_loader_kwargs"]["demo_paths"].append(variant["env_demo_path"])
    if variant.get('add_env_offpolicy_data', False):
        variant["path_loader_kwargs"]["demo_paths"].append(variant["env_offpolicy_data_path"])
    

    path_loader_kwargs = variant.get("path_loader_kwargs", {})
    # stack_obs = path_loader_kwargs.get("stack_obs", 1)
    # if stack_obs > 1:
    #     expl_env = StackObservationEnv(expl_env, stack_obs=stack_obs)
    #     eval_env = StackObservationEnv(eval_env, stack_obs=stack_obs)

    obs_dim = expl_env.observation_space.low.size
    action_dim = eval_env.action_space.low.size

    if hasattr(expl_env, 'info_sizes'):
        env_info_sizes = expl_env.info_sizes
    else:
        env_info_sizes = dict()

    # Network Initialization
    qf_kwargs = variant.get("qf_kwargs", {})

    qf1 = ParallelMlp(
        obs_dim + action_dim,
        256,
        1,
        variant['ensemble_size'],
        bias_const=0.0
    )

    qf2 = ParallelMlp(
        obs_dim + action_dim,
        256,
        1,
        variant['ensemble_size'],
        bias_const=0.0
    )

    target_qf1 = ParallelMlp(
        obs_dim + action_dim,
        256,
        1,
        variant['ensemble_size'],
        bias_const=0.0
    )

    target_qf2 = ParallelMlp(
        obs_dim + action_dim,
        256,
        1,
        variant['ensemble_size'],
        bias_const=0.0
    )



    policy_class = variant.get("policy_class", TanhGaussianPolicy)
    policy_kwargs = variant['policy_kwargs']
    # if policy_path:
    #     policy = load_local_or_remote_file(policy_path)
    # else:
    #     policy = policy_class(
    #         obs_dim=obs_dim,
    #         action_dim=action_dim,
    #         **policy_kwargs,
    #     )
    buffer_policy_path = variant.get("buffer_policy_path", False)
    if buffer_policy_path:
        buffer_policy = load_local_or_remote_file(buffer_policy_path)
    else:
        buffer_policy_class = variant.get("buffer_policy_class", policy_class)
        buffer_policy = buffer_policy_class(
            obs_dim=obs_dim,
            action_dim=action_dim,
            **variant.get("buffer_policy_kwargs", policy_kwargs),
        )
    policies = []
    for i in range(variant.get('ensemble_size', 5)):
        pol = policy_class(obs_dim=obs_dim, action_dim=action_dim, **policy_kwargs)
        pol.load_state_dict(
            torch.load('./data/awac/' + '11-12-' + env_id + '_online_policy_lr0.0003_separate_bufferFalse_seed{}/model/policy.p'.format(i*4+seed))
        )
        policies.append(pol)
    policy = ParallelGaussianPolicy(policies)

    for i in range(variant.get('ensemble_size', 5)):
        ''' Load the networks '''
        qf1_params = torch.load('./data/awac/' + '11-12-' + env_id + '_online_policy_lr0.0003_separate_bufferFalse_seed{}/model/qf1.p'.format(i*4+seed))
        qf2_params = torch.load('./data/awac/' + '11-12-' + env_id + '_online_policy_lr0.0003_separate_bufferFalse_seed{}/model/qf2.p'.format(i*4+seed))
        target_qf1_params = torch.load('./data/awac/' + '11-12-' + env_id + '_online_policy_lr0.0003_separate_bufferFalse_seed{}/model/target_qf1.p'.format(i*4+seed))
        target_qf2_params = torch.load('./data/awac/' + '11-12-' + env_id + '_online_policy_lr0.0003_separate_bufferFalse_seed{}/model/target_qf2.p'.format(i*4+seed))
        with torch.no_grad():
            qf1.h1.weights[i] = qf1_params['fc0.weight'].transpose(1, 0)
            qf1.h1.biases[i][0] = qf1_params['fc0.bias']
            qf1.h2.weights[i] = qf1_params['fc1.weight'].transpose(1, 0)
            qf1.h2.biases[i][0] = qf1_params['fc1.bias']
            qf1.output.weights[i] = qf1_params['last_fc.weight'].transpose(1, 0)
            qf1.output.biases[i][0] = qf1_params['last_fc.bias']
            qf2.h1.weights[i] = qf2_params['fc0.weight'].transpose(1, 0)
            qf2.h1.biases[i][0] = qf2_params['fc0.bias']
            qf2.h2.weights[i] = qf2_params['fc1.weight'].transpose(1, 0)
            qf2.h2.biases[i][0] = qf2_params['fc1.bias']
            qf2.output.weights[i] = qf2_params['last_fc.weight'].transpose(1, 0)
            qf2.output.biases[i][0] = qf2_params['last_fc.bias']

            target_qf1.h1.weights[i] = target_qf1_params['fc0.weight'].transpose(1, 0)
            target_qf1.h1.biases[i][0] = target_qf1_params['fc0.bias']
            target_qf1.h2.weights[i] = target_qf1_params['fc1.weight'].transpose(1, 0)
            target_qf1.h2.biases[i][0] = target_qf1_params['fc1.bias']
            target_qf1.output.weights[i] = target_qf1_params['last_fc.weight'].transpose(1, 0)
            target_qf1.output.biases[i][0] = target_qf1_params['last_fc.bias']
            target_qf2.h1.weights[i] = target_qf2_params['fc0.weight'].transpose(1, 0)
            target_qf2.h1.biases[i][0] = target_qf2_params['fc0.bias']
            target_qf2.h2.weights[i] = target_qf2_params['fc1.weight'].transpose(1, 0)
            target_qf2.h2.biases[i][0] = target_qf2_params['fc1.bias']
            target_qf2.output.weights[i] = target_qf2_params['last_fc.weight'].transpose(1, 0)
            target_qf2.output.biases[i][0] = target_qf2_params['last_fc.bias']


    eval_policy = MakeDeterministic(policy)
    eval_path_collector = MdpPathCollector(
        eval_env,
        eval_policy,
    )

    expl_policy = policy
    exploration_kwargs =  variant.get('exploration_kwargs', {})
    if exploration_kwargs:
        if exploration_kwargs.get("deterministic_exploration", False):
            expl_policy = MakeDeterministic(policy)

        exploration_strategy = exploration_kwargs.get("strategy", None)
        if exploration_strategy is None:
            pass
        elif exploration_strategy == 'ou':
            es = OUStrategy(
                action_space=expl_env.action_space,
                max_sigma=exploration_kwargs['noise'],
                min_sigma=exploration_kwargs['noise'],
            )
            expl_policy = PolicyWrappedWithExplorationStrategy(
                exploration_strategy=es,
                policy=expl_policy,
            )
        elif exploration_strategy == 'gauss_eps':
            es = GaussianAndEpsilonStrategy(
                action_space=expl_env.action_space,
                max_sigma=exploration_kwargs['noise'],
                min_sigma=exploration_kwargs['noise'],  # constant sigma
                epsilon=0,
            )
            expl_policy = PolicyWrappedWithExplorationStrategy(
                exploration_strategy=es,
                policy=expl_policy,
            )
        else:
            error


    # Replay Buffer.
    # 1. main_replay_buffer: 
    # 2. replay_buffer: 

    main_replay_buffer_kwargs=dict(
        max_replay_buffer_size=variant['replay_buffer_size'],
        env=expl_env,
    )
    replay_buffer_kwargs = dict(
        max_replay_buffer_size=variant['replay_buffer_size'],
        env=expl_env,
    )

    replay_buffer = variant.get('replay_buffer_class', EnvReplayBuffer)(
        **main_replay_buffer_kwargs,
    )
    # if variant.get('use_validation_buffer', False):
    #     train_replay_buffer = replay_buffer
    #     validation_replay_buffer = variant.get('replay_buffer_class', EnvReplayBuffer)(
    #         **main_replay_buffer_kwargs,
    #     )
    #     replay_buffer = SplitReplayBuffer(train_replay_buffer, validation_replay_buffer, 0.9)

    trainer_class = AWACEnsembleTrainer
    trainer = trainer_class(
        env=eval_env,
        policies=policies,
        qf1=qf1,
        qf2=qf2,
        target_qf1=target_qf1,
        target_qf2=target_qf2,
        buffer_policy=buffer_policy,
        ensemble_size=variant['ensemble_size'],
        **variant['trainer_kwargs']
    )
    if variant['collection_mode'] == 'online':
        expl_path_collector = MdpStepCollector(
            expl_env,
            policy,
        )
        algorithm = TorchOnlineRLAlgorithm(
            trainer=trainer,
            exploration_env=expl_env,
            evaluation_env=eval_env,
            exploration_data_collector=expl_path_collector,
            evaluation_data_collector=eval_path_collector,
            replay_buffer=replay_buffer,
            max_path_length=variant['max_path_length'],
            batch_size=variant['batch_size'],
            num_epochs=variant['num_epochs'],
            num_eval_steps_per_epoch=variant['num_eval_steps_per_epoch'],
            num_expl_steps_per_train_loop=variant['num_expl_steps_per_train_loop'],
            num_trains_per_train_loop=variant['num_trains_per_train_loop'],
            min_num_steps_before_training=variant['min_num_steps_before_training'],
        )
    else:
        expl_path_collector = MdpPathCollector(
            expl_env,
            expl_policy,
        )
        if variant['use_separate_buffer']:
            online_replay_buffer = variant.get('replay_buffer_class', EnvReplayBuffer)(
                **main_replay_buffer_kwargs,
            )
            algorithm = TorchSeparateBuffersRLAlgorithm(
                trainer=trainer,
                exploration_env=expl_env,
                evaluation_env=eval_env,
                exploration_data_collector=expl_path_collector,
                evaluation_data_collector=eval_path_collector,
                offline_replay_buffer=replay_buffer,
                online_replay_buffer=online_replay_buffer,
                max_path_length=variant['max_path_length'],
                batch_size=variant['batch_size'],
                num_epochs=variant['num_epochs'],
                num_eval_steps_per_epoch=variant['num_eval_steps_per_epoch'],
                num_expl_steps_per_train_loop=variant['num_expl_steps_per_train_loop'],
                num_trains_per_train_loop=variant['num_trains_per_train_loop'],
                min_num_steps_before_training=variant['min_num_steps_before_training'],            )
        else:
            algorithm = TorchBatchRLAlgorithm(
                trainer=trainer,
                exploration_env=expl_env,
                evaluation_env=eval_env,
                exploration_data_collector=expl_path_collector,
                evaluation_data_collector=eval_path_collector,
                replay_buffer=replay_buffer,
                max_path_length=variant['max_path_length'],
                batch_size=variant['batch_size'],
                num_epochs=variant['num_epochs'],
                num_eval_steps_per_epoch=variant['num_eval_steps_per_epoch'],
                num_expl_steps_per_train_loop=variant['num_expl_steps_per_train_loop'],
                num_trains_per_train_loop=variant['num_trains_per_train_loop'],
                min_num_steps_before_training=variant['min_num_steps_before_training'],
            )
    algorithm.to(ptu.device)






    demo_train_buffer = EnvReplayBuffer(
        **replay_buffer_kwargs,
    )
    demo_test_buffer = EnvReplayBuffer(
        **replay_buffer_kwargs,
    )





    # if variant.get("save_video", False):
    #     if variant.get("presampled_goals", None):
    #         variant['image_env_kwargs']['presampled_goals'] = load_local_or_remote_file(variant['presampled_goals']).item()

    #     def get_img_env(env):
    #         renderer = EnvRenderer(**variant["renderer_kwargs"])
    #         img_env = InsertImageEnv(GymToMultiEnv(env), renderer=renderer)

    #     image_eval_env = ImageEnv(GymToMultiEnv(eval_env), **variant["image_env_kwargs"])
    #     # image_eval_env = get_img_env(eval_env)
    #     image_eval_path_collector = ObsDictPathCollector(
    #         image_eval_env,
    #         eval_policy,
    #         observation_key="state_observation",
    #     )
    #     image_expl_env = ImageEnv(GymToMultiEnv(expl_env), **variant["image_env_kwargs"])
    #     # image_expl_env = get_img_env(expl_env)
    #     image_expl_path_collector = ObsDictPathCollector(
    #         image_expl_env,
    #         expl_policy,
    #         observation_key="state_observation",
    #     )
    #     video_func = VideoSaveFunction(
    #         image_eval_env,
    #         variant,
    #         image_expl_path_collector,
    #         image_eval_path_collector,
    #     )
    #     algorithm.post_train_funcs.append(video_func)

    if variant.get('save_paths', False):
        algorithm.post_train_funcs.append(save_paths)









    # if variant.get('load_demos', False):
    #     path_loader_class = variant.get('path_loader_class', MDPPathLoader)
    #     path_loader = path_loader_class(trainer,
    #         replay_buffer=replay_buffer,
    #         demo_train_buffer=demo_train_buffer,
    #         demo_test_buffer=demo_test_buffer,
    #         **path_loader_kwargs
    #     )
    #     path_loader.load_demos()



    if variant.get('load_env_dataset_demos', False):
        path_loader_class = variant.get('path_loader_class', HDF5PathLoader)
        path_loader = path_loader_class(trainer,
            replay_buffer=replay_buffer,
            demo_train_buffer=demo_train_buffer,
            demo_test_buffer=demo_test_buffer,
            **path_loader_kwargs
        )
        path_loader.load_demos(expl_env.get_dataset())








    # if variant.get('save_initial_buffers', False):
    #     buffers = dict(
    #         replay_buffer=replay_buffer,
    #         demo_train_buffer=demo_train_buffer,
    #         demo_test_buffer=demo_test_buffer,
    #     )
    #     buffer_path = osp.join(logger.get_snapshot_dir(), 'buffers.p')
    #     pickle.dump(buffers, open(buffer_path, "wb"))


    # if variant.get('pretrain_buffer_policy', False):
    #     trainer.pretrain_policy_with_bc(
    #         buffer_policy,
    #         replay_buffer.train_replay_buffer,
    #         replay_buffer.validation_replay_buffer,
    #         10000,
    #         label="buffer",
    #     )
    # if variant.get('pretrain_policy', False):
    #     trainer.pretrain_policy_with_bc(
    #         policy,
    #         demo_train_buffer,
    #         demo_test_buffer,
    #         trainer.bc_num_pretrain_steps,
    #     )




    # if variant.get('pretrain_rl', False):
    #     trainer.pretrain_q_with_bc_data()
   
    # if variant.get('save_pretrained_algorithm', False):
    #     p_path = osp.join(logger.get_snapshot_dir(), 'pretrain_algorithm.p')
    #     pt_path = osp.join(logger.get_snapshot_dir(), 'pretrain_algorithm.pt')
    #     data = algorithm._get_snapshot()
    #     data['algorithm'] = algorithm
    #     torch.save(data, open(pt_path, "wb"))
    #     torch.save(data, open(p_path, "wb"))




    if variant.get('train_rl', True):
        algorithm.train()
