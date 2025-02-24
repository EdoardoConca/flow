"""Runner script for single and multi-agent reinforcement learning experiments.

This script performs an RL experiment using the PPO algorithm. Choice of
hyperparameters can be seen and adjusted from the code below.

Usage
    python train.py EXP_CONFIG
"""
import argparse
from datetime import datetime
import json
import os
import sys
from time import strftime
import numpy as np
from copy import deepcopy

from flow.core.util import ensure_dir
from flow.utils.registry import env_constructor
from flow.utils.rllib import FlowParamsEncoder, get_flow_params
from flow.utils.registry import make_create_env


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
        '--algorithm', type=str, default="PPO",
        help='RL algorithm to use. Options are PPO, DQN, and A3C.')
    parser.add_argument(
        '--num_iterations', type=int, default=20,
        help='How many iterations are in a training run.')
    parser.add_argument(
        '--tune', action='store_true', default=False,
        help='Whether to perform hyperparams tuning.')
    parser.add_argument(
        '--best_hyps', action='store_true', default=False,
        help='Whether to use the best hyperparameters.')
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
        '--checkpoint_freq', type=int, default=20,
        help='How often to checkpoint.')
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


def setup_exps_rllib(flow_params,
                     n_cpus,
                     n_rollouts,
                     flags,
                     policy_graphs=None,
                     policy_mapping_fn=None,
                     policies_to_train=None):
    """Return the relevant components of an RLlib experiment.

    Parameters
    ----------
    flow_params : dict
        flow-specific parameters (see flow/utils/registry.py)
    n_cpus : int
        number of CPUs to run the experiment over
    n_rollouts : int
        number of rollouts per training iteration
    policy_graphs : dict, optional
        TODO
    policy_mapping_fn : function, optional
        TODO
    policies_to_train : list of str, optional
        TODO

    Returns
    -------
    str
        name of the training algorithm
    str
        name of the gym environment to be trained
    dict
        training configuration parameters
    """
    from ray import tune
    from ray.tune.registry import register_env
    from ray.rllib.env.group_agents_wrapper import _GroupAgentsWrapper
    try:
        from ray.rllib.agents.agent import get_agent_class
    except ImportError:
        from ray.rllib.agents.registry import get_agent_class

    horizon = flow_params['env'].horizon
    alg_run = flags.algorithm.upper()
    computed_batch_size = horizon * n_rollouts

    if alg_run == "PPO":

        agent_cls = get_agent_class(alg_run)
        config = deepcopy(agent_cls._default_config)

        if flags.best_hyps:
            bets_params_path = "/home/edoardo_reply/ray_results/PPO/PPO_CustomNormalizedMultiAgentAccelPOEnv-v1_9e1245e8_2025-02-24_14-45-32mcli7plc/params.json"

            with open(bets_params_path, "r") as f:
                best_params = json.load(f)

            config.update(best_params)

            flags.tune = False
            
        else:
            config.update({
                "num_workers": n_cpus,
                "train_batch_size": computed_batch_size ,
                "gamma": 0.999,  # discount rate
                "model": {"fcnet_hiddens": [32, 32, 32]},
                "use_gae": True,
                "lambda": 0.97,
                "kl_target": 0.02,
                "num_sgd_iter": 10,
                "horizon": horizon,
            })

        if flags.tune:
            config.update({
                "lambda": tune.sample_from(lambda _: np.float32(np.random.choice([0.8, 0.9, 0.95, 0.97]))),
                "lr": tune.loguniform(1e-5, 1e-3),  
                "train_batch_size": tune.choice([1024, 2048, 4096, computed_batch_size]),
                "gamma": tune.sample_from(lambda _: np.float32(np.random.choice([0.98, 0.99, 0.999]))),
                "entropy_coeff": tune.sample_from(lambda _: np.float32(np.random.choice([0.01, 0.02, 0.05]))),
                "kl_target": tune.sample_from(lambda _: np.float32(np.random.choice([0.01, 0.02, 0.03]))),
                "num_sgd_iter": tune.choice([5, 10]),
            })

    elif alg_run == "DDPG":

        # Ottieni la classe dell'agente DDPG
        agent_cls = get_agent_class(alg_run)
        config = deepcopy(agent_cls._default_config)

        if flags.best_hyps:
            bets_params_path = "/home/edoardo_reply/ray_results/DDPG/DDPG_CustomNormalizedMultiAgentAccelPOEnv-v1_19b5796d_2025-02-24_12-15-330r4iv2e0/params.json"

            with open(bets_params_path, "r") as f:
                best_params = json.load(f)

            config.update(best_params)
            flags.tune = False
        else:
            config.update({
                "num_workers": n_cpus,
                "gamma": 0.99,  
                "tau": 0.01,  
                "horizon": horizon,
                "actor_hiddens": [128, 128],  
                "critic_hiddens": [128, 128],  
                "actor_lr": 1e-4,  
                "critic_lr": 1e-3,  
                "train_batch_size": 1024,  
                "buffer_size": 100000,  
                "exploration_noise_type": "ou",  
                "exploration_ou_theta": 0.15,  
                "exploration_ou_sigma": 0.2,  
                "exploration_ou_noise_scale": 0.1,  
            })
            
        
        if flags.tune:
            config.update({
                "exploration_ou_theta": tune.uniform(0.1, 0.3),
                "exploration_ou_sigma": tune.uniform(0.05, 0.3),
                "exploration_ou_noise_scale": tune.uniform(0.01, 0.2),
                "actor_lr": tune.loguniform(1e-5, 1e-3),
                "critic_lr": tune.loguniform(1e-5, 1e-3),
                "train_batch_size": tune.choice([256, 512, 1024]),
                "buffer_size": tune.choice([50000, 100000, 200000]),
            })
    
    elif alg_run == "A3C":

        agent_cls = get_agent_class(alg_run)
        config = deepcopy(agent_cls._default_config)
        if flags.best_hyps:
            bets_params_path = "/home/edoardo_reply/ray_results/A3C/A3C_CustomNormalizedMultiAgentAccelPOEnv-v1_e6b90d9d_2025-02-24_00-50-037th8kfte/params.json"

            with open(bets_params_path, "r") as f:
                best_params = json.load(f)

            config.update(best_params)
            flags.tune = False
        else:
            config.update({
                "num_workers": n_cpus,  
                "sample_batch_size": 10,  
                "train_batch_size": computed_batch_size,  
                "gamma": 0.99,  
                "lambda": 1.0, 
                "grad_clip": 40.0,  
                "lr": 0.0001,  
                "lr_schedule": None,  
                "vf_loss_coeff": 0.5,  
                "entropy_coeff": 0.01,  
                "min_iter_time_s": 5,  
                "sample_async": True,  
                "use_pytorch": False,  
            })

        if flags.tune:
            config.update({
                "lambda": tune.choice([0.8, 0.9, 0.95, 0.97]),
                "lr": tune.loguniform(1e-5, 1e-3),
                "train_batch_size": tune.choice([1024, 2048, 4096, computed_batch_size]),
                "gamma": tune.choice([0.98, 0.99, 0.999]),
                "entropy_coeff": tune.choice([0.01, 0.02, 0.05]),
                "vf_loss_coeff": tune.choice([0.5, 0.7, 1.0]),
            })
    else:
        sys.exit("We only support PPO, DDPG and A3C, right now.")
        

    # **Callbacks**
    def on_episode_start(info):
        """Inizializza le metriche per ogni episodio."""
        episode = info["episode"]
        episode.user_data["fuel_consumption"] = []  
        episode.user_data["collisions"] = 0  
        episode.user_data["num_cars"] = []  
        episode.user_data["avg_accel_human"] = []  
        episode.user_data["avg_accel_avs"] = []  
        episode.user_data["avg_speed_human"] = []  
        episode.user_data["avg_speed_avs"] = []  
        episode.user_data["avg_speed"] = []  

    def on_episode_step(info):
        """Registra i dati a ogni passo dell'episodio."""
        episode = info["episode"]
        env = info["env"].get_unwrapped()[0]

        veh_ids = [veh_id for veh_id in env.k.vehicle.get_ids() if env.k.vehicle.get_speed(veh_id) >= 0]
        rl_ids = [veh_id for veh_id in env.k.vehicle.get_rl_ids() if env.k.vehicle.get_speed(veh_id) >= 0]
        human_ids = [veh_id for veh_id in env.k.vehicle.get_human_ids() if env.k.vehicle.get_speed(veh_id) >= 0]

        # Velocità media
        if veh_ids:
            episode.user_data["avg_speed"].append(np.mean([env.k.vehicle.get_speed(veh_id) for veh_id in veh_ids]))
        if rl_ids:
            episode.user_data["avg_speed_avs"].append(np.mean([env.k.vehicle.get_speed(veh_id) for veh_id in rl_ids]))
        if human_ids:
            episode.user_data["avg_speed_human"].append(np.mean([env.k.vehicle.get_speed(veh_id) for veh_id in human_ids]))

        # Numero di veicoli
        episode.user_data["num_cars"].append(len(veh_ids))

        # Accelerazione media
        if human_ids:
            accel_values_human = [
                np.abs((env.k.vehicle.get_speed(veh_id) - env.k.vehicle.get_previous_speed(veh_id)) / env.sim_step)
                for veh_id in human_ids if veh_id in env.k.vehicle.previous_speeds.keys()
            ]
            if accel_values_human:
                episode.user_data["avg_accel_human"].append(np.mean(accel_values_human))

        if rl_ids:
            accel_values_avs = [
                np.abs((env.k.vehicle.get_speed(veh_id) - env.k.vehicle.get_previous_speed(veh_id)) / env.sim_step)
                for veh_id in rl_ids if veh_id in env.k.vehicle.previous_speeds.keys()
            ]
            if accel_values_avs:
                episode.user_data["avg_accel_avs"].append(np.mean(accel_values_avs))

        # Consumo di carburante
        episode.user_data["fuel_consumption"].extend(
            [env.k.vehicle.get_fuel_consumption(veh_id) for veh_id in veh_ids]
        )

        # Collisioni
        if env.k.simulation.check_collision():
            episode.user_data["collisions"] += 1

    def on_episode_end(info):
        """Calcola le metriche finali per l'episodio e stampa i risultati."""
        episode = info["episode"]

        print("DEBUG: user_data contenuto prima della media")
        for k, v in episode.user_data.items():
            print(f"Key: {k}, Type: {type(v)}, Length: {len(v) if isinstance(v, list) else 'N/A'}")

        def clean_and_mean(data):
            """Appiattisce i dati e calcola la media ignorando valori NaN e non numerici."""
            if isinstance(data, list):
                flat_data = [float(item) for item in data if isinstance(item, (int, float)) and not np.isnan(item)]
                return np.mean(flat_data) if flat_data else 0  
            return float(data) if isinstance(data, (int, float)) else 0  

        # **Usa direttamente i nomi originali senza _mean**
        for k in episode.user_data:
            try:
                episode.custom_metrics[k] = clean_and_mean(episode.user_data[k])
            except Exception as e:
                print(f"Errore su {k}: {str(e)}, sto ignorando.")
                episode.custom_metrics[k] = 0

        # **Evita duplicati come `_mean_mean`**
        episode.custom_metrics["fuel_consumption"] = clean_and_mean(episode.user_data["fuel_consumption"])
        episode.custom_metrics["total_collisions"] = episode.user_data.get("collisions", 0)
        episode.custom_metrics["avg_speed"] = clean_and_mean(episode.user_data["avg_speed"])
        episode.custom_metrics["avg_speed_avs"] = clean_and_mean(episode.user_data["avg_speed_avs"])
        episode.custom_metrics["avg_speed_human"] = clean_and_mean(episode.user_data["avg_speed_human"])
        episode.custom_metrics["num_cars"] = clean_and_mean(episode.user_data["num_cars"])
        episode.custom_metrics["avg_accel_avs"] = clean_and_mean(episode.user_data["avg_accel_avs"])
        episode.custom_metrics["avg_accel_human"] = clean_and_mean(episode.user_data["avg_accel_human"])

    def on_train_result(info):
        """Registra le metriche globali per il training."""
        trainer = info["trainer"]
        trainer.workers.foreach_worker(
            lambda ev: ev.foreach_env(
                lambda env: env.set_iteration_num()))

    # **Registra i callback**
    config['callbacks'] = {
        "on_episode_start": tune.function(on_episode_start),
        "on_episode_step": tune.function(on_episode_step),
        "on_episode_end": tune.function(on_episode_end),
        "on_train_result": tune.function(on_train_result)
    }

    # Salva i parametri per il replay
    flow_json = json.dumps(
        flow_params, cls=FlowParamsEncoder, sort_keys=True, indent=4)
    config['env_config']['flow_params'] = flow_json
    config['env_config']['run'] = alg_run

    # Configurazione multi-agente
    if policy_graphs is not None:
        config['multiagent'].update({'policies': policy_graphs})
    if policy_mapping_fn is not None:
        config['multiagent'].update(
            {'policy_mapping_fn': tune.function(policy_mapping_fn)})
    if policies_to_train is not None:
        config['multiagent'].update({'policies_to_train': policies_to_train})

    create_env, gym_name = make_create_env(params=flow_params)

    # Registra l'ambiente su RLlib
    register_env(gym_name, create_env)
    return alg_run, gym_name, config


def train_rllib(submodule, flags):
    """Train policies using the PPO algorithm in RLlib."""
    import ray
    from ray.tune.schedulers import ASHAScheduler
    from ray.tune import run_experiments
    import pytz
    from ray import tune


    flow_params = submodule.flow_params
    n_cpus = submodule.N_CPUS
    n_rollouts = submodule.N_ROLLOUTS
    policy_graphs = getattr(submodule, "POLICY_GRAPHS", None)
    policy_mapping_fn = getattr(submodule, "policy_mapping_fn", None)
    policies_to_train = getattr(submodule, "policies_to_train", None)

    alg_run, gym_name, config = setup_exps_rllib(
        flow_params, n_cpus, n_rollouts,flags,
        policy_graphs, policy_mapping_fn, policies_to_train)

    config['num_workers'] = flags.num_cpus
    config['env'] = gym_name

    horizon = flow_params['env'].horizon
    num_iterations = flags.num_iterations

    if flags.algorithm.upper() in ["PPO", "A3C"]:
        total_timesteps = horizon * n_rollouts * num_iterations
    elif flags.algorithm.upper() == "DDPG":
        if flags.tune:
            total_timesteps = None  #config["train_batch_size"] is a sample_from object, so we can't access the value directly
        else:
            total_timesteps = config["train_batch_size"] * num_iterations  
    else:
        raise ValueError(f"We only support PPO, DDPG and A3C, right now.")

    # create a custom string that makes looking at the experiment names easier
    def trial_str_creator(trial):
        return "{}_{}".format(trial.trainable_name, trial.experiment_tag)
    
    # **Hyperparameter Tuning con Ray Tune**
    if flags.tune:
        stop_criteria = {
            "training_iteration": 50,  
            "episode_reward_mean": 7000,  
        }

        scheduler = ASHAScheduler(
            metric="episode_reward_mean",
            mode="max",
            max_t=50,
            grace_period=5,
            reduction_factor=2
        )

        ray.init(num_cpus=n_cpus + 1, object_store_memory=200 * 1024 * 1024)

        tune.run(
            alg_run,  
            config=config,
            num_samples=10,  
            scheduler=scheduler,  
            stop=stop_criteria,  
        )

    else:
        ray.init(num_cpus=n_cpus + 1, object_store_memory=200 * 1024 * 1024)
        
        exp_config = {
            "run": alg_run,
            "env": gym_name,
            "config": {
                **config
            },
            "checkpoint_freq": flags.checkpoint_freq,
            "checkpoint_at_end": True,
            'trial_name_creator': trial_str_creator,
            "max_failures": 0,
            "stop": {
                "training_iteration": 50,
            },
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
