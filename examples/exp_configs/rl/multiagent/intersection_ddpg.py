from flow.utils.registry import make_create_env
from ray.rllib.agents.ddpg.ddpg_policy import DDPGTFPolicy
from examples.exp_configs.experiment_configs_ddpg import get_flow_params

N_CPUS=3
N_ROLLOUTS=50

# start sumo visualization
flow_params = get_flow_params()


# Create the environment
create_env, gym_name = make_create_env(params=flow_params)

test_env = create_env()
obs_space = test_env.observation_space
act_space = test_env.action_space


def gen_policy():
    """Generate a policy in RLlib."""
    return DDPGTFPolicy, obs_space, act_space, {}

# Setup PG with an ensemble of `num_policies` different policy graphs
POLICY_GRAPHS = {'av': gen_policy()}


def policy_mapping_fn(_):
    """Map a policy in RLlib."""
    return 'av'


POLICIES_TO_TRAIN = ['av']

