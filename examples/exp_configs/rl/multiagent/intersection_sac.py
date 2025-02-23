from flow.utils.registry import make_create_env
from ray.rllib.agents.sac.sac_policy import SACTFPolicy
from examples.exp_configs.experiment_configs_sac import get_flow_params
from gym.envs.registration import registry

# Numero di CPU e rollout per training
N_CPUS = 3
N_ROLLOUTS = 50


# Carica i parametri dell'esperimento
flow_params = get_flow_params()

create_env, gym_name = make_create_env(params=flow_params)
test_env = create_env()  # Crea SOLO per ottenere obs_space e act_space
obs_space = test_env.observation_space
act_space = test_env.action_space

def gen_policy():
    """Genera una policy SAC in RLlib."""
    return SACTFPolicy, obs_space, act_space, {}

# Configura la policy in RLlib
POLICY_GRAPHS = {'av': gen_policy()}

def policy_mapping_fn(_):
    """Mappa la policy agli agenti in RLlib."""
    return 'av'

# Definisce quali policy verranno addestrate
POLICIES_TO_TRAIN = ['av']