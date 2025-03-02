"""Visualizer for rllib experiments.

Attributes
----------
EXAMPLE_USAGE : str
    Example call to the function, which is
    ::

        python ./visualizer_rllib.py /tmp/ray/result_dir 1

parser : ArgumentParser
    Command-line argument parser
"""

import argparse
import gym
import numpy as np
import os
import sys
import time

import ray
try:
    from ray.rllib.agents.agent import get_agent_class
except ImportError:
    from ray.rllib.agents.registry import get_agent_class
from ray.tune.registry import register_env

from flow.core.util import emission_to_csv
from flow.utils.registry import make_create_env
from flow.utils.rllib import get_flow_params
from flow.utils.rllib import get_rllib_config
from flow.utils.rllib import get_rllib_pkl
sys.path.append(os.path.abspath("./examples/exp_configs"))


EXAMPLE_USAGE = """
example usage:
    python ./visualizer_rllib.py /ray_results/experiment_dir/result_dir 1

Here the arguments are:
1 - the path to the simulation results
2 - the number of the checkpoint
"""


def visualizer_rllib(args):
    """Visualizer for RLlib experiments.

    This function takes args (see function create_parser below for
    more detailed information on what information can be fed to this
    visualizer), and renders the experiment associated with it.
    """
    result_dir = args.result_dir if args.result_dir[-1] != '/' \
        else args.result_dir[:-1]

    config = get_rllib_config(result_dir)

    # check if we have a multiagent environment but in a
    # backwards compatible way
    if config.get('multiagent', {}).get('policies', None):
        multiagent = True
        pkl = get_rllib_pkl(result_dir)
        config['multiagent'] = pkl['multiagent']
    else:
        multiagent = False

    # Run on only one cpu for rendering purposes
    config['num_workers'] = 1

    flow_params = get_flow_params(config)
    sim_params = flow_params['sim']

    # Determine agent and checkpoint
    config_run = config['env_config']['run'] if 'run' in config['env_config'] \
        else None
    if args.run and config_run:
        if args.run != config_run:
            print('visualizer_rllib.py: error: run argument '
                  + '\'{}\' passed in '.format(args.run)
                  + 'differs from the one stored in params.json '
                  + '\'{}\''.format(config_run))
            sys.exit(1)
    if args.run:
        agent_cls = get_agent_class(args.run)
    elif config_run:
        agent_cls = get_agent_class(config_run)
    else:
        print('visualizer_rllib.py: error: could not find flow parameter '
              '\'run\' in params.json, '
              'add argument --run to provide the algorithm or model used '
              'to train the results\n e.g. '
              'python ./visualizer_rllib.py /tmp/ray/result_dir 1 --run PPO')
        sys.exit(1)

    sim_params.restart_instance = True
    dir_path = os.path.dirname(os.path.realpath(__file__))
    emission_path = '{0}/test_time_rollout/'.format(dir_path)
    sim_params.emission_path = emission_path if args.gen_emission else None

    # pick your rendering mode
    if args.render_mode == 'sumo_web3d':
        sim_params.num_clients = 2
        sim_params.render = False
    elif args.render_mode == 'drgb':
        sim_params.render = 'drgb'
        sim_params.pxpm = 4
    elif args.render_mode == 'sumo_gui':
        sim_params.render = False  # will be set to True below
    elif args.render_mode == 'no_render':
        sim_params.render = False
    if args.save_render:
        if args.render_mode != 'sumo_gui':
            sim_params.render = 'drgb'
            sim_params.pxpm = 4
        sim_params.save_render = True

    # Create and register a gym+rllib env
    create_env, env_name = make_create_env(params=flow_params, version=0)
    register_env(env_name, create_env)

    # check if the environment is a single or multiagent environment, and
    # get the right address accordingly
    # single_agent_envs = [env for env in dir(flow.envs)
    #                      if not env.startswith('__')]

    # if flow_params['env_name'] in single_agent_envs:
    #     env_loc = 'flow.envs'
    # else:
    #     env_loc = 'flow.envs.multiagent'

    # Start the environment with the gui turned on and a path for the
    # emission file
    env_params = flow_params['env']
    env_params.restart_instance = False
    if args.evaluate:
        env_params.evaluate = True

    # lower the horizon if testing
    if args.horizon:
        config['horizon'] = args.horizon
        env_params.horizon = args.horizon

    # create the agent that will be used to compute the actions
    agent = agent_cls(env=env_name, config=config)
    checkpoint = result_dir + '/checkpoint_' + args.checkpoint_num
    checkpoint = checkpoint + '/checkpoint-' + args.checkpoint_num
    agent.restore(checkpoint)

    if hasattr(agent, "local_evaluator") and \
            os.environ.get("TEST_FLAG") != 'True':
        env = agent.local_evaluator.env
    else:
        env = gym.make(env_name)

    if args.render_mode == 'sumo_gui':
        env.sim_params.render = True  # set to True after initializing agent and env

    if multiagent:
        rets = {}
        # map the agent id to its policy
        policy_map_fn = config['multiagent']['policy_mapping_fn']
        for key in config['multiagent']['policies'].keys():
            rets[key] = []
    else:
        rets = []

    if config.get('model', {}).get('use_lstm', False):
        use_lstm = True
        if multiagent:
            state_init = {}
            # map the agent id to its policy
            policy_map_fn = config['multiagent']['policy_mapping_fn']
            size = config['model']['lstm_cell_size']
            for key in config['multiagent']['policies'].keys():
                state_init[key] = [np.zeros(size, np.float32),
                                   np.zeros(size, np.float32)]
        else:
            state_init = [
                np.zeros(config['model']['lstm_cell_size'], np.float32),
                np.zeros(config['model']['lstm_cell_size'], np.float32)
            ]
    else:
        use_lstm = False

    # if restart_instance, don't restart here because env.reset will restart later
    if not sim_params.restart_instance:
        env.restart_simulation(sim_params=sim_params, render=sim_params.render)

    # Simulate and collect metrics
    final_outflows, final_inflows, mean_speed, std_speed = [], [], [], []
    total_collisions, fuel_consumptions, avg_accelerations, avg_num_cars = [], [], [], []
    mean_rl_speed, mean_idm_speed = [], []

    for i in range(args.num_rollouts):
        rl_vel, idm_vel, vel, fuel, accelerations, num_cars= [], [], [], [], [], []
        collisions = 0
        state = env.reset()
        env.k.simulation.simulation_step()


        if multiagent:
            ret = {key: [0] for key in rets.keys()}
        else:
            ret = 0

        for _ in range(env_params.horizon):
            # Compute actions
            if multiagent:
                action = {}
                for agent_id in state.keys():
                    if use_lstm:
                        action[agent_id], state_init[agent_id], logits = \
                            agent.compute_action(
                            state[agent_id], state=state_init[agent_id],
                            policy_id=policy_map_fn(agent_id))
                    else:
                        action[agent_id] = agent.compute_action(state[agent_id], policy_id=policy_map_fn(agent_id))
            else:
                action = agent.compute_action(state)

            state, reward, done, _ = env.step(action)

            if multiagent:
                for actor, rew in reward.items():
                    ret[policy_map_fn(actor)][0] += rew
            else:
                ret += reward

            vehicles = env.unwrapped.k.vehicle
            veh_ids = vehicles.get_ids()
            idm_ids = vehicles.get_human_ids()
            rl_ids = vehicles.get_rl_ids()

            speeds = vehicles.get_speed(veh_ids)
            rl_speeds = vehicles.get_speed(rl_ids)
            idm_speeds = vehicles.get_speed(idm_ids)
            if speeds:
                vel.append(np.nanmean(speeds))  # Media velocità
            if rl_speeds:
                rl_vel.append(np.nanmean(rl_speeds))
            if idm_speeds:
                idm_vel.append(np.nanmean(idm_speeds))

            # Number of vehicles
            num_cars.append(len(veh_ids))

            # Fuel consumption
            fuel_consumed = vehicles.get_fuel_consumption(veh_ids)
            if fuel_consumed:
                fuel.append(np.nanmean(fuel_consumed))  # Media consumo carburante

            # Accelerazione media dei veicoli
            accel_values = [
                np.abs((vehicles.get_speed(veh_id) - vehicles.get_previous_speed(veh_id)) / env.sim_step)
                for veh_id in veh_ids if veh_id in vehicles.previous_speeds.keys()
            ]
            if accel_values:
                accelerations.append(np.nanmean(accel_values))  # Media accelerazione

            # Collision detection
            simulation = env.unwrapped.k.simulation
            if simulation.check_collision():
                collisions += 1
            
            if multiagent and done['__all__']:
                break
            if not multiagent and done:
                break

        if multiagent:
            for key in ret.keys():
                rets[key].append(ret[key][0])
        else:
            rets.append(ret)

        # Memorizza le metriche raccolte
        final_outflows.append(vehicles.get_outflow_rate(500))
        final_inflows.append(vehicles.get_inflow_rate(500))

        throughput_efficiency = (final_outflows[-1] / final_inflows[-1]) if final_inflows[-1] > 1e-5 else 0

        avg_num_cars.append(np.mean(num_cars))
        total_collisions.append(collisions)
        fuel_consumptions.append(np.mean(fuel) if fuel else 0)
        avg_accelerations.append(np.mean(accelerations) if accelerations else 0)

        mean_speed.append(np.mean(vel))
        std_speed.append(np.std(vel))
        mean_rl_speed.append(np.mean(rl_vel))
        mean_idm_speed.append(np.mean(idm_vel))

        print(f'Round {i}, Return: {ret}, Collisions: {collisions}, Throughput Efficiency: {throughput_efficiency:.3f}')

        # Save emission data if required
        if env.simulator == "traci":
            env.k.simulation.save_emission(run_id=i)

    # Stampa le metriche finali
    print("\n==== Summary of results ====")
    if multiagent:
        for agent_id, rew in rets.items():
            print(f"Avg Return for {agent_id}: {np.mean(rew):.2f} ")
    else:
        print(f"Avg Return: {np.mean(rets):.2f}")

    print(f"Avg Speed (m/s): {np.mean(mean_speed)}")
    print(f"Avg RL Speed (m/s): {np.mean(mean_rl_speed)}")
    print(f"Avg IDM Speed (m/s): {np.mean(mean_idm_speed)}")
    print(f"Avg Fuel Consumption: {np.mean(fuel_consumptions)}")
    print(f"Avg Acceleration: {np.mean(avg_accelerations)}")
    print(f"Avg number of vehicles: {np.mean(avg_num_cars)}")
    print(f"Avg Collisions: {np.mean(total_collisions)}")
    print(f"Avg Collision: {np.sum(total_collisions)}")
    print(f"Avg Outflow: {np.mean(final_outflows)}")
    print(f"Throughput Efficiency: {np.mean(throughput_efficiency)}")


    # terminate the environment
    env.unwrapped.terminate()

    # if prompted, convert the emission file into a csv file
    if args.gen_emission:
        time.sleep(0.1)

        dir_path = os.path.dirname(os.path.realpath(__file__))
        emission_filename = '{0}-emission.xml'.format(env.network.name)

        emission_path = \
            '{0}/test_time_rollout/{1}'.format(dir_path, emission_filename)

        # convert the emission file into a csv file
        emission_to_csv(emission_path)

        # print the location of the emission csv file
        emission_path_csv = emission_path[:-4] + ".csv"
        print("\nGenerated emission file at " + emission_path_csv)

        # delete the .xml version of the emission file
        os.remove(emission_path)


def create_parser():
    """Create the parser to capture CLI arguments."""
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description='[Flow] Evaluates a reinforcement learning agent '
                    'given a checkpoint.',
        epilog=EXAMPLE_USAGE)

    # required input parameters
    parser.add_argument(
        'result_dir', type=str, help='Directory containing results')
    parser.add_argument('checkpoint_num', type=str, help='Checkpoint number.')

    # optional input parameters
    parser.add_argument(
        '--run',
        type=str,
        help='The algorithm or model to train. This may refer to '
             'the name of a built-on algorithm (e.g. RLLib\'s DQN '
             'or PPO), or a user-defined trainable function or '
             'class registered in the tune registry. '
             'Required for results trained with flow-0.2.0 and before.')
    parser.add_argument(
        '--num_rollouts',
        type=int,
        default=1,
        help='The number of rollouts to visualize.')
    parser.add_argument(
        '--gen_emission',
        action='store_true',
        help='Specifies whether to generate an emission file from the '
             'simulation')
    parser.add_argument(
        '--evaluate',
        action='store_true',
        help='Specifies whether to use the \'evaluate\' reward '
             'for the environment.')
    parser.add_argument(
        '--render_mode',
        type=str,
        default='sumo_gui',
        help='Pick the render mode. Options include sumo_web3d, '
             'rgbd and sumo_gui')
    parser.add_argument(
        '--save_render',
        action='store_true',
        help='Saves a rendered video to a file. NOTE: Overrides render_mode '
             'with pyglet rendering.')
    parser.add_argument(
        '--horizon',
        type=int,
        help='Specifies the horizon.')
    return parser


if __name__ == '__main__':
    parser = create_parser()
    args = parser.parse_args()
    ray.init(num_cpus=1)
    visualizer_rllib(args)
