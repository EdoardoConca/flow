<img src="docs/img/square_logo.png" align="right" width="25%"/>

[![Build Status](https://travis-ci.com/flow-project/flow.svg?branch=master)](https://travis-ci.com/flow-project/flow)
[![Docs](https://readthedocs.org/projects/flow/badge)](http://flow.readthedocs.org/en/latest/)
[![Coverage Status](https://coveralls.io/repos/github/flow-project/flow/badge.svg?branch=master)](https://coveralls.io/github/flow-project/flow?branch=master)
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/flow-project/flow/binder)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](https://github.com/flow-project/flow/blob/master/LICENSE.md)

# Flow

[Flow](https://flow-project.github.io/) is a computational framework for deep RL and control experiments for traffic microsimulation.

See [our website](https://flow-project.github.io/) for more information on the application of Flow to several mixed-autonomy traffic scenarios. Other [results and videos](https://sites.google.com/view/ieee-tro-flow/home) are available as well.

# More information

- [Documentation](https://flow.readthedocs.org/en/latest/)
- [Installation instructions](http://flow.readthedocs.io/en/latest/flow_setup.html)
- [Tutorials](https://github.com/flow-project/flow/tree/master/tutorials)
- [Binder Build (beta)](https://mybinder.org/v2/gh/flow-project/flow/binder)

# Technical questions

If you have a bug, please report it. Otherwise, join the [Flow Users group](https://join.slack.com/t/flow-users/shared_invite/enQtODQ0NDYxMTQyNDY2LTY1ZDVjZTljM2U0ODIxNTY5NTQ2MmUxMzYzNzc5NzU4ZTlmNGI2ZjFmNGU4YjVhNzE3NjcwZTBjNzIxYTg5ZmY) on Slack!  

# Getting involved

We welcome your contributions.

- Please report bugs and improvements by submitting [GitHub issue](https://github.com/flow-project/flow/issues).
- Submit your contributions using [pull requests](https://github.com/flow-project/flow/pulls). Please use [this template](https://github.com/flow-project/flow/blob/master/.github/PULL_REQUEST_TEMPLATE.md) for your pull requests.

# Citing Flow

If you use Flow for academic research, you are highly encouraged to cite our paper:

C. Wu, A. Kreidieh, K. Parvate, E. Vinitsky, A. Bayen, "Flow: Architecture and Benchmarking for Reinforcement Learning in Traffic Control," CoRR, vol. abs/1710.05465, 2017. [Online]. Available: https://arxiv.org/abs/1710.05465

If you use the benchmarks, you are highly encouraged to cite our paper:

Vinitsky, E., Kreidieh, A., Le Flem, L., Kheterpal, N., Jang, K., Wu, F., ... & Bayen, A. M,  Benchmarks for reinforcement learning in mixed-autonomy traffic. In Conference on Robot Learning (pp. 399-409). Available: http://proceedings.mlr.press/v87/vinitsky18a.html

# Contributors

Flow is supported by the [Mobile Sensing Lab](http://bayen.eecs.berkeley.edu/) at UC Berkeley and Amazon AWS Machine Learning research grants. The contributors are listed in [Flow Team Page](https://flow-project.github.io/team.html).


# MAS-project-conca-2324

  


## Project Overview

  

This project implements a multi-agent reinforcement learning (MARL) system for autonomous vehicle coordination at unsignalized intersections using Flow, a framework for integrating SUMO with reinforcement learning libraries like RLlib.

  

The project extends the standard Flow environment by:

  
- Implementing a **custom  SUMO network** for the mixed traffic unsignalized intersection, that can be found at `mas_project/flow/network/intersection.py`

- Creating **custom environments** for multi-agent training, `CustomNormalizedMultiAgentAccelPOEnv` and `VariantNormalizedMultiAgentAccelPOEnv` at `mas_project/flow/env/multiagent/intersection.py`

- Modifying **training script**, `examples/train.py` to track important metrics and train different algorithms. Morevover added the possibility to run hyperparameter tuning.

- Adding an **environment without RL**, `CustomTestEnv`  ,to serve as a baseline, at `mas_project/flow/envs/intersection_base.py`

- Adding the **throughput_reward** at `flow/core/rewards.py` 

- Creating a **new visualizer** to display important key metrics to evaluate training, at `flow/visualize/visualizer_rllib_new.py` 

- Modifying the `Experiment` class at `flow/core/experiment.py` to display important key metrics to evaluate the Non RL environment `CustomTestEnv`.

-  The project includes dedicated experiment configuration files for three reinforcement learning models: **A3C, PPO, and DDPG**. Each model has a corresponding configuration script defining the environment setup, traffic inflow, vehicle parameters, and RL settings.
```
mas_project/ 
├── flow/ 
├── examples/ 
│ 	├── train.py 							# Script to train the RL models 
│ 	├── simulate.py 						# Script to run simulation without RL 
│ 	├── exp_configs/ 
│ 	│ 	├── experiment_configs_a3c.py 		# Configuration for A3C 
│ 	│ 	├── experiment_configs_ppo.py 		# Configuration for PPO 
│ 	│ 	├── experiment_configs_ddpg.py 		# Configuration for DDPG 
│ 	│ 	├── non_rl/
│ 	│ 	│ 	│ 	├── intersection.py 		# No RL configuration
│ 	│ 	├── rl/ 
│ 	│ 	│ 	├── multiagent/ 
│ 	│ 	│ 	│ 	├── intersection.py 		# Multi-agent PPO RL configuration
│ 	│ 	│ 	│ 	├── intersection_a3c.py 	# Multi-agent A3C RL configuration
│ 	│ 	│ 	│ 	├── intersection_ddpg.py 	# Multi-agent DDPG RL configuration
│── README.md
```

## Running the Experiments

  

1. Training an RL Model

  

To train an RL model (A3C, PPO, or DDPG) on the custom intersection environment:

  

```

python examples/train.py intersection_a3c --algorithm a3c --best_hyps --num_iterations 50 --num_cpus 3

```

  

2. Visualizing Training Results

  

To visualize a trained model:

  

```

python flow/visualize/visualizer_rllib_new.py ~/ray_results/RESULT_DIR --checkpoint_num 50 --horizon 1500 --num_rollouts 5 --render_mode no_render
```

  

3. Running the Environment Without RL

  

To simulate the intersection without reinforcement learning (baseline scenario):

  

```

python examples/simulate.py intersection --num_runs 5 --horizon 1500 --no_render

```

  

## Project Structure

```
mas_project/ 
├── examples/ 
│ 	├── train.py 							
│ 	├── simulate.py 						 
│ 	├── exp_configs/ 
│ 	│ 	├── experiment_configs_a3c.py 		
│ 	│ 	├── experiment_configs_ppo.py 		 
│ 	│ 	└── experiment_configs_ddpg.py 
│ 	│ 	├── non_rl/
│ 	│ 	│ 	│ 	└── intersection.py 		 
│ 	│ 	├── rl/ 
│ 	│ 	│ 	├── multiagent/ 
│ 	│ 	│ 	│ 	├── intersection.py 		
│ 	│ 	│ 	│ 	├── intersection_a3c.py 	
│ 	│ 	│ 	│ 	└── intersection_ddpg.py 	
├── flow/ 
│   ├── envs/
│   │   ├── multiagent/
│   │   │     └── intersection.py
│   │   ├── visualize/
│   │   │     └── visualizer_rllib_new.py
│   │   ├── intersection_base.py
│── README.md
└── requirements.txt		
```

  

## Key Features

  

- Multi-Agent Reinforcement Learning for traffic coordination.

- Custom reward functions optimizing throughput and safety.

- Baseline environment for comparison with RL-based approaches.

- Custom metrics tracking for collision rate, fuel usage, speed, throughput and acceleration.

- Multiple RL algorithms supported (A3C, PPO, DDPG).

- Easy configuration and extensibility with modular experiment files.

  

## Future Improvements

 - Fine-tune RL models for better generalization across different traffic scenarios.

- Implement more advanced reward functions specifically for the chosen alghoritm.

- Extend the project to multi-intersection coordination.

- Incorporating Vehicle-to-Vehicle (V2V) Communication

## Author

  

Edoardo Conca