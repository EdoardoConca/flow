from flow.envs.base import Env
from flow.core.rewards import desired_velocity, penalize_standstill, rl_forward_progress
from flow.core.rewards import min_delay, punish_rl_lane_changes
from gym.spaces.box import Box
import numpy as np

ADDITIONAL_ENV_PARAMS = {
    "max_accel": 3,  # max acceleration (m/s^2)
    "max_decel": 3,  # max deceleration (m/s^2)
    "target_velocity": 20,  # desired velocity (m/s)
}

class CustomTestEnv(Env):
    """Test environment used to run simulations in the absence of autonomy.

    Required from env_params
        None

    Optional from env_params
        reward_fn : A reward function which takes an an input the environment
        class and returns a real number.

    States
        States are an empty list.

    Actions
        No actions are provided to any RL agent.

    Rewards
        The reward is zero at every step.

    Termination
        A rollout is terminated if the time horizon is reached or if two
        vehicles collide into one another.
    """

    @property
    def action_space(self):
        """See parent class."""
        return Box(low=0, high=0, shape=(0,), dtype=np.float32)

    @property
    def observation_space(self):
        """See parent class."""
        return Box(low=0, high=0, shape=(0,), dtype=np.float32)

    def _apply_rl_actions(self, rl_actions):
        return

    def compute_reward(self, rl_actions, **kwargs):
        """See parent class."""
        if "reward_fn" in self.env_params.additional_params:
            return self.env_params.additional_params["reward_fn"](self)
        else:
            return 0

    def get_state(self, **kwargs):
        """Restituisce la velocità media dei veicoli nella rete."""
        veh_ids = self.k.vehicle.get_ids()
        
        if len(veh_ids) == 0:
            return np.array([0])  # Nessun veicolo attivo

        speeds = [self.k.vehicle.get_speed(veh_id) for veh_id in veh_ids if self.k.vehicle.get_speed(veh_id) >= 0]

        if len(speeds) == 0:
            return np.array([0])  # Nessuna velocità disponibile
        else:
            avg_speed = np.mean(speeds)
            return np.array([avg_speed])  # Velocità media della rete
