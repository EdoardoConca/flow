"""Runner script for single and multi-agent reinforcement learning experiments.

This script performs an RL experiment using the PPO algorithm. Choice of
hyperparameters can be seen and adjusted from the code below.

Usage
    python train.py EXP_CONFIG
"""
import argparse
import json
import os
import sys
from time import strftime
from copy import deepcopy

from flow.core.util import ensure_dir
from flow.utils.registry import env_constructor
from flow.utils.rllib import FlowParamsEncoder, get_flow_params
from flow.utils.registry import make_create_env

import gym
import numpy as np
from gym.envs.registration import register
from gym.envs.registration import registry

from copy import deepcopy

import flow.envs
from flow.core.params import InitialConfig
from flow.core.params import TrafficLightParams

import os
import ray


def make_create_env(params, version=0, render=None):
    """Create a parametrized flow environment compatible with OpenAI gym."""
    exp_tag = params["exp_tag"]

    if isinstance(params["env_name"], str):
        print("""Passing of strings for env_name will be deprecated.
        Please pass the Env instance instead.""")
        base_env_name = params["env_name"]
    else:
        base_env_name = params["env_name"].__name__

    # Genera un nome univoco per l'ambiente
    env_name = f"{base_env_name}-v{version}"

    # Verifica se l'ambiente è già registrato
    if env_name in registry.env_specs:
        print(f"L'ambiente {env_name} è già registrato. Verrà utilizzato l'ambiente esistente.")
    else:
        # Se l'ambiente non è registrato, procedi con la registrazione
        if isinstance(params["network"], str):
            print("""Passing of strings for network will be deprecated.
            Please pass the Network instance instead.""")
            module = __import__("flow.networks", fromlist=[params["network"]])
            network_class = getattr(module, params["network"])
        else:
            network_class = params["network"]

        env_params = params['env']
        net_params = params['net']
        initial_config = params.get('initial', InitialConfig())
        traffic_lights = params.get("tls", TrafficLightParams())

        def create_env(*_):
            sim_params = deepcopy(params['sim'])
            vehicles = deepcopy(params['veh'])

            network = network_class(
                name=exp_tag,
                vehicles=vehicles,
                net_params=net_params,
                initial_config=initial_config,
                traffic_lights=traffic_lights,
            )

            # Accetta un nuovo tipo di rendering se non è impostato su None
            sim_params.render = render or sim_params.render

            # Verifica se l'ambiente è single-agent o multi-agent
            single_agent_envs = [env for env in dir(flow.envs)
                               if not env.startswith('__')]

            if isinstance(params["env_name"], str):
                if params['env_name'] in single_agent_envs:
                    env_loc = 'flow.envs'
                else:
                    env_loc = 'flow.envs.multiagent'
                entry_point = env_loc + ':{}'.format(params["env_name"])
            else:
                entry_point = params["env_name"].__module__ + ':' + params["env_name"].__name__

            # Registra l'ambiente solo se non è già registrato
            if env_name not in registry.env_specs:
                register(
                    id=env_name,
                    entry_point=entry_point,
                    kwargs={
                        "env_params": env_params,
                        "sim_params": sim_params,
                        "network": network,
                        "simulator": params['simulator']
                    })
                print(f"Registrato l'ambiente: {env_name}")
            else:
                print(f"L'ambiente {env_name} è già registrato. Verrà utilizzato l'ambiente esistente.")

            return gym.envs.make(env_name)

        return create_env, env_name

    # Se l'ambiente è già registrato, restituisci una funzione create_env che usa l'ambiente esistente
    def create_env(*_):
        return gym.envs.make(env_name)

    return create_env, env_name


def parse_args(args):
    """Parse training options user can specify in command line.

    Returns
    -------
    argparse.Namespace
        the output parser object
    """
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="Parse argument used when running a Flow simulation.",
        epilog="python train.py EXP_CONFIG")

    # required input parameters
    parser.add_argument(
        'exp_config', type=str,
        help='Name of the experiment configuration file, as located in '
             'exp_configs/rl/singleagent or exp_configs/rl/multiagent.')

    # optional input parameters
    parser.add_argument(
        '--rl_trainer', type=str, default="rllib",
        help='the RL trainer to use. either rllib or Stable-Baselines')

    parser.add_argument(
        '--num_cpus', type=int, default=1,
        help='How many CPUs to use')
    parser.add_argument(
        '--num_steps', type=int, default=5000,
        help='How many total steps to perform learning over')
    parser.add_argument(
        '--rollout_size', type=int, default=1000,
        help='How many steps are in a training batch.')
    parser.add_argument(
        '--checkpoint_path', type=str, default=None,
        help='Directory with checkpoint to restore training from.')

    return parser.parse_known_args(args)[0]


def run_model_stablebaseline(flow_params,
                             num_cpus=1,
                             rollout_size=50,
                             num_steps=50):
    """Run the model for num_steps if provided.

    Parameters
    ----------
    flow_params : dict
        flow-specific parameters
    num_cpus : int
        number of CPUs used during training
    rollout_size : int
        length of a single rollout
    num_steps : int
        total number of training steps
    The total rollout length is rollout_size.

    Returns
    -------
    stable_baselines.*
        the trained model
    """
    from stable_baselines.common.vec_env import DummyVecEnv, SubprocVecEnv
    from stable_baselines import PPO2

    if num_cpus == 1:
        constructor = env_constructor(params=flow_params, version=0)()
        # The algorithms require a vectorized environment to run
        env = DummyVecEnv([lambda: constructor])
    else:
        env = SubprocVecEnv([env_constructor(params=flow_params, version=i)
                             for i in range(num_cpus)])

    train_model = PPO2('MlpPolicy', env, verbose=1, n_steps=rollout_size)
    train_model.learn(total_timesteps=num_steps)
    return train_model


def setup_exps_rllib(flow_params, n_cpus, n_rollouts,
                     policy_graphs=None, policy_mapping_fn=None, policies_to_train=None):
    """Return the relevant components of an RLlib SAC experiment."""
    from ray import tune
    from ray.tune.registry import register_env
    from gym.envs.registration import registry
    try:
        from ray.rllib.agents.agent import get_agent_class
    except ImportError:
        from ray.rllib.agents.registry import get_agent_class

    horizon = flow_params['env'].horizon
    alg_run = "SAC"

    agent_cls = get_agent_class(alg_run)
    config = deepcopy(agent_cls._default_config)

    config.update({
        "num_workers": n_cpus,
        "train_batch_size": 256,
        "learning_starts": 1500,
        "buffer_size": int(1e6),
        "tau": 5e-3,
        "gamma": 0.99,
        "n_step": 1,
        "target_entropy": "auto",
        "optimization": {
            "actor_learning_rate": 3e-4,
            "critic_learning_rate": 3e-4,
            "entropy_learning_rate": 3e-4,
        },
        "target_network_update_freq": 0,
        "use_state_preprocessor": False,
        "policy": "GaussianLatentSpacePolicy",
        "Q_model": {
            "hidden_activation": "relu",
            "hidden_layer_sizes": (256, 256),
        },
        "policy_model": {
            "hidden_activation": "relu",
            "hidden_layer_sizes": (256, 256),
        },
        "prioritized_replay": False,
        "num_gpus": 0,
        "num_cpus_per_worker": 1,
        "evaluation_interval": 1,
        "evaluation_num_episodes": 1,
        "evaluation_config": {"exploration_enabled": False},
        "compress_observations": False,
    })


    # save the flow params for replay
    flow_json = json.dumps(
        flow_params, cls=FlowParamsEncoder, sort_keys=True, indent=4)
    config['env_config']['flow_params'] = flow_json
    config['env_config']['run'] = alg_run

    # multiagent configuration
    if policy_graphs is not None:
        config['multiagent'].update({'policies': policy_graphs})
    if policy_mapping_fn is not None:
        config['multiagent'].update(
            {'policy_mapping_fn': tune.function(policy_mapping_fn)})
    if policies_to_train is not None:
        config['multiagent'].update({'policies_to_train': policies_to_train})

    create_env, gym_name = make_create_env(params=flow_params)

    # Register as rllib env
    register_env(gym_name, create_env)
    return alg_run, gym_name, config


def train_rllib(submodule, flags):
    """Train policies using the SAC algorithm in RLlib."""
    import ray
    from ray.tune import run_experiments

    flow_params = submodule.flow_params
    n_cpus = submodule.N_CPUS
    n_rollouts = submodule.N_ROLLOUTS
    policy_graphs = getattr(submodule, "POLICY_GRAPHS", None)
    policy_mapping_fn = getattr(submodule, "policy_mapping_fn", None)
    policies_to_train = getattr(submodule, "policies_to_train", None)

    alg_run, gym_name, config = setup_exps_rllib(
        flow_params, n_cpus, n_rollouts,
        policy_graphs, policy_mapping_fn, policies_to_train)
    

    ray.init(
        num_cpus=n_cpus + 1,
        ignore_reinit_error=True,
        object_store_memory=200 * 1024 * 1024, 
    )

    exp_config = {
        "run": alg_run,
        "env": gym_name,
        "config": {**config},
        "checkpoint_freq": 20,
        "checkpoint_at_end": True,
        "max_failures": 0,
        "stop": {"training_iteration": flags.num_steps},
    }

    if flags.checkpoint_path is not None:
        exp_config['restore'] = flags.checkpoint_path

    run_experiments({flow_params["exp_tag"]: exp_config})


def train_h_baselines(env_name, args, multiagent):
    """Train policies using SAC and TD3 with h-baselines."""
    from hbaselines.algorithms import OffPolicyRLAlgorithm
    from hbaselines.utils.train import parse_options, get_hyperparameters

    # Get the command-line arguments that are relevant here
    args = parse_options(description="", example_usage="", args=args)

    # the base directory that the logged data will be stored in
    base_dir = "training_data"

    for i in range(args.n_training):
        # value of the next seed
        seed = args.seed + i

        # The time when the current experiment started.
        now = strftime("%Y-%m-%d-%H:%M:%S")

        # Create a save directory folder (if it doesn't exist).
        dir_name = os.path.join(base_dir, '{}/{}'.format(args.env_name, now))
        ensure_dir(dir_name)

        # Get the policy class.
        if args.alg == "TD3":
            if multiagent:
                from hbaselines.multi_fcnet.td3 import MultiFeedForwardPolicy
                policy = MultiFeedForwardPolicy
            else:
                from hbaselines.fcnet.td3 import FeedForwardPolicy
                policy = FeedForwardPolicy
        elif args.alg == "SAC":
            if multiagent:
                from hbaselines.multi_fcnet.sac import MultiFeedForwardPolicy
                policy = MultiFeedForwardPolicy
            else:
                from hbaselines.fcnet.sac import FeedForwardPolicy
                policy = FeedForwardPolicy
        else:
            raise ValueError("Unknown algorithm: {}".format(args.alg))

        # Get the hyperparameters.
        hp = get_hyperparameters(args, policy)

        # Add the seed for logging purposes.
        params_with_extra = hp.copy()
        params_with_extra['seed'] = seed
        params_with_extra['env_name'] = args.env_name
        params_with_extra['policy_name'] = policy.__name__
        params_with_extra['algorithm'] = args.alg
        params_with_extra['date/time'] = now

        # Add the hyperparameters to the folder.
        with open(os.path.join(dir_name, 'hyperparameters.json'), 'w') as f:
            json.dump(params_with_extra, f, sort_keys=True, indent=4)

        # Create the algorithm object.
        alg = OffPolicyRLAlgorithm(
            policy=policy,
            env="flow:{}".format(env_name),
            eval_env="flow:{}".format(env_name) if args.evaluate else None,
            **hp
        )

        # Perform training.
        alg.learn(
            total_steps=args.total_steps,
            log_dir=dir_name,
            log_interval=args.log_interval,
            eval_interval=args.eval_interval,
            save_interval=args.save_interval,
            initial_exploration_steps=args.initial_exploration_steps,
            seed=seed,
        )


def train_stable_baselines(submodule, flags):
    """Train policies using the PPO algorithm in stable-baselines."""
    from stable_baselines.common.vec_env import DummyVecEnv
    from stable_baselines import PPO2

    flow_params = submodule.flow_params
    # Path to the saved files
    exp_tag = flow_params['exp_tag']
    result_name = '{}/{}'.format(exp_tag, strftime("%Y-%m-%d-%H:%M:%S"))

    # Perform training.
    print('Beginning training.')
    model = run_model_stablebaseline(
        flow_params, flags.num_cpus, flags.rollout_size, flags.num_steps)

    # Save the model to a desired folder and then delete it to demonstrate
    # loading.
    print('Saving the trained model!')
    path = os.path.realpath(os.path.expanduser('~/baseline_results'))
    ensure_dir(path)
    save_path = os.path.join(path, result_name)
    model.save(save_path)

    # dump the flow params
    with open(os.path.join(path, result_name) + '.json', 'w') as outfile:
        json.dump(flow_params, outfile,
                  cls=FlowParamsEncoder, sort_keys=True, indent=4)

    # Replay the result by loading the model
    print('Loading the trained model and testing it out!')
    model = PPO2.load(save_path)
    flow_params = get_flow_params(os.path.join(path, result_name) + '.json')
    flow_params['sim'].render = True
    env = env_constructor(params=flow_params, version=0)()
    # The algorithms require a vectorized environment to run
    eval_env = DummyVecEnv([lambda: env])
    obs = eval_env.reset()
    reward = 0
    for _ in range(flow_params['env'].horizon):
        action, _states = model.predict(obs)
        obs, rewards, dones, info = eval_env.step(action)
        reward += rewards
    print('the final reward is {}'.format(reward))


def main(args):
    """Perform the training operations."""
    # Parse script-level arguments (not including package arguments).
    flags = parse_args(args)

    # Import relevant information from the exp_config script.
    module = __import__(
        "exp_configs.rl.singleagent", fromlist=[flags.exp_config])
    module_ma = __import__(
        "exp_configs.rl.multiagent", fromlist=[flags.exp_config])

    # Import the sub-module containing the specified exp_config and determine
    # whether the environment is single agent or multi-agent.
    if hasattr(module, flags.exp_config):
        submodule = getattr(module, flags.exp_config)
        multiagent = False
    elif hasattr(module_ma, flags.exp_config):
        submodule = getattr(module_ma, flags.exp_config)
        assert flags.rl_trainer.lower() in ["rllib", "h-baselines"], \
            "Currently, multiagent experiments are only supported through "\
            "RLlib. Try running this experiment using RLlib: " \
            "'python train.py EXP_CONFIG'"
        multiagent = True
    else:
        raise ValueError("Unable to find experiment config.")

    # Perform the training operation.
    if flags.rl_trainer.lower() == "rllib":
        train_rllib(submodule, flags)
    elif flags.rl_trainer.lower() == "stable-baselines":
        train_stable_baselines(submodule, flags)
    elif flags.rl_trainer.lower() == "h-baselines":
        train_h_baselines(flags.exp_config, args, multiagent)
    else:
        raise ValueError("rl_trainer should be either 'rllib', 'h-baselines', "
                         "or 'stable-baselines'.")


if __name__ == "__main__":
    main(sys.argv[1:])
