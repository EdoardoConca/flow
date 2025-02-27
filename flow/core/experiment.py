"""Contains an experiment class for running simulations."""
from flow.utils.registry import make_create_env
from datetime import datetime
import logging
import time
import numpy as np


class Experiment:
    """
    Class for systematically running simulations in any supported simulator.

    This class acts as a runner for a network and environment. In order to use
    it to run a network and environment in the absence of a method specifying
    the actions of RL agents in the network, type the following:

        >>> from flow.envs import Env
        >>> flow_params = dict(...)  # see the examples in exp_config
        >>> exp = Experiment(flow_params)  # for some experiment configuration
        >>> exp.run(num_runs=1)

    If you wish to specify the actions of RL agents in the network, this may be
    done as follows:

        >>> rl_actions = lambda state: 0  # replace with something appropriate
        >>> exp.run(num_runs=1, rl_actions=rl_actions)

    Finally, if you would like to plot and visualize your results, this
    class can generate csv files from emission files produced by sumo.
    Finally, if you would like to like to plot and visualize your results, this
    class can generate csv files from emission files produced by sumo. These
    files will contain the speeds, positions, edges, etc... of every vehicle
    in the network at every time step.

    In order to ensure that the simulator constructs an emission file, set the
    ``emission_path`` attribute in ``SimParams`` to some path.

        >>> from flow.core.params import SimParams
        >>> flow_params['sim'] = SimParams(emission_path="./data")

    Once you have included this in your environment, run your Experiment object
    as follows:

        >>> exp.run(num_runs=1, convert_to_csv=True)

    After the experiment is complete, look at the "./data" directory. There
    will be two files, one with the suffix .xml and another with the suffix
    .csv. The latter should be easily interpretable from any csv reader (e.g.
    Excel), and can be parsed using tools such as numpy and pandas.

    Attributes
    ----------
    custom_callables : dict < str, lambda >
        strings and lambda functions corresponding to some information we want
        to extract from the environment. The lambda will be called at each step
        to extract information from the env and it will be stored in a dict
        keyed by the str.
    env : flow.envs.Env
        the environment object the simulator will run
    """

    def __init__(self, flow_params, custom_callables=None):
        """Instantiate the Experiment class.

        Parameters
        ----------
        flow_params : dict
            flow-specific parameters
        custom_callables : dict < str, lambda >
            strings and lambda functions corresponding to some information we
            want to extract from the environment. The lambda will be called at
            each step to extract information from the env and it will be stored
            in a dict keyed by the str.
        """
        self.custom_callables = custom_callables or {}

        # Get the env name and a creator for the environment.
        create_env, _ = make_create_env(flow_params)

        # Create the environment.
        self.env = create_env()

        logging.info(" Starting experiment {} at {}".format(
            self.env.network.name, str(datetime.utcnow())))

        logging.info("Initializing environment.")

    def run(self, num_runs, rl_actions=None, convert_to_csv=False):
        """Run the given network for a set number of runs.

        Parameters
        ----------
        num_runs : int
            number of runs the experiment should perform
        rl_actions : method, optional
            maps states to actions to be performed by the RL agents (if
            there are any)
        convert_to_csv : bool
            Specifies whether to convert the emission file created by sumo
            into a csv file

        Returns
        -------
        info_dict : dict < str, Any >
            contains returns, average speed per step, collisions, acceleration,
            fuel consumption, and throughput efficiency.
        """
        num_steps = self.env.env_params.horizon

        # raise an error if convert_to_csv is set to True but no emission
        # file will be generated, to avoid getting an error at the end of the
        # simulation
        if convert_to_csv and self.env.sim_params.emission_path is None:
            raise ValueError(
                'The experiment was run with convert_to_csv set '
                'to True, but no emission file will be generated. If you wish '
                'to generate an emission file, you should set the parameter '
                'emission_path in the simulation parameters to a valid path.')

        # used to store
        info_dict = {
            "returns": [],
            "velocities": [],
            "accelerations": [],
            "fuel_consumption": [],
            "collisions": [],
            "outflows": [],
            "throughput_efficiency": [],
        }
        info_dict.update({key: [] for key in self.custom_callables.keys()})

        if rl_actions is None:
            def rl_actions(*_):
                return None

        # time profiling information
        t = time.time()
        times = []

        for i in range(num_runs):
            ret = 0
            vel = []
            accs = []
            fuel_cons = []
            collisions = 0
            custom_vals = {key: [] for key in self.custom_callables.keys()}
            state = self.env.reset()

            for j in range(num_steps):
                t0 = time.time()
                state, reward, done, _ = self.env.step(rl_actions(state))
                t1 = time.time()
                times.append(1 / (t1 - t0))

                veh_ids = self.env.k.vehicle.get_ids()
                
                # Collect metrics
                if veh_ids:
                    vel.append(np.nanmean(self.env.k.vehicle.get_speed(veh_ids)))
                else:
                    vel.append(0)  
                accel_values = [
                    np.abs((self.env.k.vehicle.get_speed(veh_id) - self.env.k.vehicle.get_previous_speed(veh_id)) / self.env.sim_step)
                    for veh_id in veh_ids if veh_id in self.env.k.vehicle.previous_speeds.keys()
                ]
                accs.append(np.nanmean(accel_values) if accel_values else 0)
                if veh_ids:
                    fuel_cons.append(np.nanmean(self.env.k.vehicle.get_fuel_consumption(veh_ids)))
                else:
                    fuel_cons.append(0) 
                
                # Check for collisions
                if self.env.k.simulation.check_collision():
                    collisions += 1

                ret += reward

                # Compute custom callables
                for (key, lambda_func) in self.custom_callables.items():
                    custom_vals[key].append(lambda_func(self.env))

                if done:
                    break

            # Compute outflow and throughput efficiency
            outflow = self.env.k.vehicle.get_outflow_rate(500)
            inflow = self.env.k.vehicle.get_inflow_rate(500)
            throughput_efficiency = outflow / inflow if inflow > 1e-5 else 0

            # Store results
            info_dict["returns"].append(ret)
            info_dict["velocities"].append(np.nanmean(vel))
            info_dict["accelerations"].append(np.nanmean(accs))
            info_dict["fuel_consumption"].append(np.nanmean(fuel_cons))
            info_dict["collisions"].append(collisions)
            info_dict["outflows"].append(outflow)
            info_dict["throughput_efficiency"].append(throughput_efficiency)

            for key in custom_vals.keys():
                info_dict[key].append(np.mean(custom_vals[key]))

            print(f"Round {i}, return: {ret}, collisions: {collisions}, throughput efficiency: {throughput_efficiency}")

            # Save emission data if required
            if self.env.simulator == "traci":
                self.env.k.simulation.save_emission(run_id=i)

        # Print the averages for all stored variables
        for key in info_dict.keys():
            print(f"Average {key}: {np.mean(info_dict[key])}")

        print("Total time:", time.time() - t)
        print("Steps/second:", np.mean(times))
        self.env.terminate()

        return info_dict
