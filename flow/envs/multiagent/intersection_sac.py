from flow.envs.multiagent import MultiEnv
from flow.core.rewards import desired_velocity, penalize_standstill, rl_forward_progress
from flow.core.rewards import min_delay, penalize_near_standstill, punish_rl_lane_changes, energy_consumption
from flow.core import rewards
from gym.spaces import Box
import numpy as np
import pyarrow
import pickle


ADDITIONAL_ENV_PARAMS = {
    "max_accel": 3,
    "max_decel": 3,
    "target_velocity": 20,
}

class SACCustomNormalizedMultiAgentAccelPOEnv(MultiEnv):
    def __init__(self, env_params, sim_params, network, simulator='traci'):
        for p in ['max_accel', 'max_decel', 'target_velocity']:
            if p not in env_params.additional_params:
                raise KeyError(f'Environment parameter "{p}" not supplied')

        self.leader = []
        self.follower = []
        self.timer = 1

        super().__init__(env_params, sim_params, network, simulator)

    @property
    def action_space(self):
        return Box(
            low=-abs(self.env_params.additional_params["max_decel"]),
            high=self.env_params.additional_params["max_accel"],
            shape=(1,),
            dtype=np.float32
        )

    @property
    def observation_space(self):
        return Box(low=-5, high=5, shape=(6,), dtype=np.float32)

    def _apply_rl_actions(self, rl_actions):
        for veh_id, action in rl_actions.items():
            if not isinstance(action, np.ndarray):
                action = np.array([action], dtype=np.float32)
            self.k.vehicle.apply_acceleration(veh_id, action)

    def compute_reward(self, rl_actions, **kwargs):
        desired_vel_reward = desired_velocity(self, fail=kwargs['fail'])
        standstill_penalty = penalize_standstill(self, gain=1.0)
        forward_progress_reward = rl_forward_progress(self, gain=0.1)
        penalize_lane_change = punish_rl_lane_changes(self)
        minimum_delay = min_delay(self)
        combined_reward = desired_vel_reward + standstill_penalty + forward_progress_reward + minimum_delay + penalize_lane_change
        cost1 = rewards.desired_velocity(self, fail=kwargs["fail"])
        cost2 = 0
        t_min = 1
        for rl_id in self.k.vehicle.get_rl_ids():
            lead_id = self.k.vehicle.get_leader(rl_id)
            if lead_id not in ["", None] and self.k.vehicle.get_speed(rl_id) > 0:
                t_headway = max(self.k.vehicle.get_headway(rl_id) / self.k.vehicle.get_speed(rl_id), 0)
                cost2 += min((t_headway - t_min) / t_min, 0)

        total_reward = (combined_reward * 0.1) + cost1 + cost2
        return {key: total_reward for key in self.k.vehicle.get_rl_ids()}

    def get_state(self):
        self.leader = []
        self.follower = []
        obs = {}

        max_speed = self.k.network.max_speed()
        max_length = self.k.network.length()

        for rl_id in self.k.vehicle.get_rl_ids():
            this_pos = self.k.vehicle.get_x_by_id(rl_id)
            this_speed = self.k.vehicle.get_speed(rl_id)
            lead_id = self.k.vehicle.get_leader(rl_id)
            follower = self.k.vehicle.get_follower(rl_id)

            if lead_id in ["", None]:
                lead_speed = max_speed
                lead_head = max_length
            else:
                self.leader.append(lead_id)
                lead_speed = self.k.vehicle.get_speed(lead_id)
                lead_head = self.k.vehicle.get_x_by_id(lead_id) - this_pos - self.k.vehicle.get_length(rl_id)

            if follower in ["", None]:
                follow_speed = 0
                follow_head = max_length
            else:
                self.follower.append(follower)
                follow_speed = self.k.vehicle.get_speed(follower)
                follow_head = self.k.vehicle.get_headway(follower)

            normalized_obs = np.array([
                this_pos / max_length,
                this_speed / max_speed,
                (lead_speed - this_speed) / max_speed,
                lead_head / max_length,
                (this_speed - follow_speed) / max_speed,
                follow_head / max_length
            ], dtype=np.float32)

            normalized_obs = np.clip(normalized_obs, -5, 5)
            obs[rl_id] = np.asarray(normalized_obs, dtype=np.float32)

        return obs

    def additional_command(self):
        for veh_id in self.leader + self.follower:
            self.k.vehicle.set_observed(veh_id)

    def reset(self):
        self.leader = []
        self.follower = []
        return super().reset()
    
    def set_iteration_num(self):
        if not hasattr(self, 'iteration'):
            self.iteration = 0
        self.iteration += 1
        print(f"Iteration updated: {self.iteration}")