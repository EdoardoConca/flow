from flow.envs.multiagent import MultiEnv
from flow.core.rewards import desired_velocity, penalize_standstill, rl_forward_progress
from flow.core.rewards import min_delay, punish_rl_lane_changes
from flow.core import rewards
from gym.spaces import Box
import numpy as np


ADDITIONAL_ENV_PARAMS = {
    # maximum acceleration for autonomous vehicles, in m/s^2
    "max_accel": 3,
    # maximum deceleration for autonomous vehicles, in m/s^2
    "max_decel": 3,
    # desired velocity for all vehicles in the network, in m/s
    "target_velocity": 20,
}

class CustomNormalizedMultiAgentAccelPOEnv(MultiEnv):
    """Custom multi-agent acceleration environment with combined rewards.

    This environment extends the MultiAgentAccelPOEnv to include additional
    reward components such as penalizing standstill and encouraging forward
    progress for RL vehicles.

    Required from env_params:
    * max_accel: maximum acceleration for autonomous vehicles, in m/s^2
    * max_decel: maximum deceleration for autonomous vehicles, in m/s^2
    * target_velocity: desired velocity for all vehicles in the network, in m/s

    States:
        The observation of each agent (i.e., each autonomous vehicle) consists
        of the speeds and bumper-to-bumper headways of the vehicles immediately
        preceding and following the autonomous vehicle, as well as the absolute
        position and ego speed of the autonomous vehicles. This results in a
        state space of size 6 for each agent.

    Actions:
        The action space for each agent consists of a scalar bounded
        acceleration for each autonomous vehicle. In order to ensure safety,
        these actions are further bounded by failsafes provided by the
        simulator at every time step.

    Rewards:
        The reward function is a combination of:
        1. The two-norm of the distance of the speed of the vehicles in the
           network from the "target_velocity" term.
        2. A penalty for vehicles that are at a standstill.
        3. A reward for RL vehicles making forward progress.

    Termination:
        A rollout is terminated if the time horizon is reached or if two
        vehicles collide into one another.
    """

    def __init__(self, env_params, sim_params, network, simulator='traci'):
        for p in ['max_accel', 'max_decel', 'target_velocity']:
            if p not in env_params.additional_params:
                raise KeyError(f'Environment parameter "{p}" not supplied')

        self.leader = []
        self.follower = []
        self.timer = 1
        self.num_training_iters = 0

        super().__init__(env_params, sim_params, network, simulator)

    @property
    def action_space(self):
        """See class definition."""
        return Box(
            low=-abs(self.env_params.additional_params["max_decel"]),
            high=self.env_params.additional_params["max_accel"],
            shape=(1,),
            dtype=np.float32
        )

    @property
    def observation_space(self):
        """See class definition."""
        return Box(low=-5, high=5, shape=(6,), dtype=np.float32)

    def _apply_rl_actions(self, rl_actions):
        for veh_id, action in rl_actions.items():
            if not isinstance(action, np.ndarray):
                action = np.array([action], dtype=np.float32)
            self.k.vehicle.apply_acceleration(veh_id, action)

    def compute_reward(self, rl_actions, **kwargs):
        """Compute the reward for each agent using combined reward functions."""
        # Compute the desired velocity reward
        desired_vel_reward = desired_velocity(self, fail=kwargs['fail'])

        # Penalize standstill
        standstill_penalty = penalize_standstill(self, gain=1.0)

        # Reward RL vehicles for forward progress
        forward_progress_reward = rl_forward_progress(self, gain=0.1)

        #Reward Vehicle for minimum Lane Changes
        penalize_lane_change = punish_rl_lane_changes(self)

        #Reward Vehicle for small Travel Time
        minimum_delay = min_delay(self)

        # Combine the rewards with Threshold
        combined_reward = desired_vel_reward + standstill_penalty + forward_progress_reward + minimum_delay + penalize_lane_change

        cost1 = rewards.desired_velocity(self, fail=kwargs["fail"])

        # penalize small time headways
        cost2 = 0
        t_min = 1  # smallest acceptable time headway
        for rl_id in self.k.vehicle.get_rl_ids():
            lead_id = self.k.vehicle.get_leader(rl_id)
            if lead_id not in ["", None] \
                    and self.k.vehicle.get_speed(rl_id) > 0:
                t_headway = max(
                    self.k.vehicle.get_headway(rl_id) /
                    self.k.vehicle.get_speed(rl_id), 0)
                cost2 += min((t_headway - t_min) / t_min, 0)

        total_reward = (combined_reward * 0.1) + cost1 + cost2

        # Reward is shared by all agents
        return {key: total_reward for key in self.k.vehicle.get_rl_ids()}

    def get_state(self):
        """Return the state of the environment."""
        self.leader = []
        self.follower = []
        obs = {}

        # Normalizing constants
        max_speed = self.k.network.max_speed()
        max_length = self.k.network.length()

        for rl_id in self.k.vehicle.get_rl_ids():
            this_pos = self.k.vehicle.get_x_by_id(rl_id)
            this_speed = self.k.vehicle.get_speed(rl_id)
            lead_id = self.k.vehicle.get_leader(rl_id)
            follower = self.k.vehicle.get_follower(rl_id)

            if lead_id in ["", None]:
                # If leader is not visible, use default values
                lead_speed = max_speed
                lead_head = max_length
            else:
                self.leader.append(lead_id)
                lead_speed = self.k.vehicle.get_speed(lead_id)
                lead_head = self.k.vehicle.get_x_by_id(lead_id) \
                    - self.k.vehicle.get_x_by_id(rl_id) \
                    - self.k.vehicle.get_length(rl_id)

            if follower in ["", None]:
                # If follower is not visible, use default values
                follow_speed = 0
                follow_head = max_length
            else:
                self.follower.append(follower)
                follow_speed = self.k.vehicle.get_speed(follower)
                follow_head = self.k.vehicle.get_headway(follower)

            # Normalize observations
            normalized_obs = np.array([
                this_pos / max_length,  # Normalize position
                this_speed / max_speed,  # Normalize speed
                (lead_speed - this_speed) / max_speed,  # Normalize relative speed
                lead_head / max_length,  # Normalize headway
                (this_speed - follow_speed) / max_speed,  # Normalize relative speed
                follow_head / max_length  # Normalize headway
            ], dtype=np.float32)

            # Clip observations to ensure they are within bounds
            normalized_obs = np.clip(normalized_obs, -5, 5)

            # Add the normalized observation to the dictionary
            obs[rl_id] = np.asarray(normalized_obs, dtype=np.float32)

        return obs

    def additional_command(self):
        """See parent class.

        This method defines which vehicles are observed for visualization
        purposes.
        """
        # Specify observed vehicles
        for veh_id in self.leader + self.follower:
            self.k.vehicle.set_observed(veh_id)

    def reset(self):
        """See parent class.
        """
        self.leader = []
        self.follower = []
        return super().reset()
    
    def set_iteration_num(self):
        self.num_training_iters += 1
