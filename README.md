# Offline-to-Online Reinforcement Learning via Balanced Replay and Pessimistic Q-Ensemble
This repository is no longer maintained.

## Installation
This codebase is based on rlkit (https://github.com/vitchyr/rlkit) and D4RL (https://github.com/rail-berkeley/d4rl). <br/>
To set up the environment and run an example experiment:

1. Create conda virtual environment. 
```
$ cd rlkit
$ conda env create -f environment/linux-gpu-env.yml
```

2. Add this repo directory to your `PYTHONPATH` environment variable or simply run:
```
$ conda activate rlkit
(rlkit) $ pip install -e .
```

3. Update pip and torch, and install D4RL
```
(rlkit) $ pip install -U pip
(rlkit) $ pip install torch==1.4.0
(rlkit) $ cd ..
(rlkit) $ cd d4rl
(rlkit) $ pip install -e .
```
## Run a demo experiment
Fine-tune an offline halfcheetah agent
```
(rlkit) $ cd ..
(rlkit) $ cd rlkit
(rlkit) $ python examples/ours.py --env_id halfcheetah-medium-v0 --policy_lr 3e-4 --first_epoch_multiplier 5 --init_online_fraction 0.75 --online_buffer_size 250000 --seed 0
```
