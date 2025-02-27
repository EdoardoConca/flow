from flow.core.params import SumoParams, EnvParams, InitialConfig, NetParams, VehicleParams, TrafficLightParams, SumoCarFollowingParams, SumoLaneChangeParams
from flow.controllers.routing_controllers import MinicityRouter
from flow.controllers import IDMController, RLController
from flow.networks.intersection import Intersection
from flow.envs.multiagent.intersection import CustomNormalizedMultiAgentAccelPOEnv, VariantNormalizedMultiAgentAccelPOEnv
from flow.core.params import InFlows
from flow.envs.test import TestEnv


FLOW_RATE = 200 #vehicles per hour 50
# target velocity
TARGET_VELOCITY = 20
# maximum acceleration for autonomous vehicles, in m/s^2
MAX_ACCEL = 3
# maximum deceleration for autonomous vehicles, in m/s^2
MAX_DECEL = 3
# time horizon of a single rollout
HORIZON = 1500
# number of rollouts per training iteration
N_ROLLOUTS = 40
# number of parallel workers
N_CPUS = 3


def get_flow_params(rl_flag=True):
    """Definisce i parametri di Flow per la simulazione."""

    # Configurazione dei veicoli
    vehicles = VehicleParams()

    # Veicoli che percorrono la griglia orizzontalmente
    
    vehicles.add(
    veh_id="human",
    acceleration_controller=(IDMController, {}),
    routing_controller=(MinicityRouter, {}),
    car_following_params=SumoCarFollowingParams(
        speed_mode="obey_safe_speed",
        decel=1.5,
    ),
    lane_change_params=SumoLaneChangeParams(
        lane_change_mode="no_lc_safe",

    ),
    num_vehicles=0)

    vehicles.add(
        veh_id="rl",
        acceleration_controller=(RLController, {}),
        routing_controller=(MinicityRouter, {}),
        car_following_params=SumoCarFollowingParams(
            speed_mode="obey_safe_speed",
            accel=MAX_ACCEL,
            decel=MAX_DECEL,
        ),
        num_vehicles=0)

    inflow = InFlows()
    for edge in ["L0", "L2", "L4", "L6"]:
        inflow.add(
            veh_type="human",
            edge=edge,
            vehs_per_hour= FLOW_RATE,
            depart_lane="random",
            depart_speed=10)
        inflow.add(
            veh_type="rl",
            edge=edge,
            vehs_per_hour= FLOW_RATE,
            depart_lane="random",
            depart_speed=5)

    

    # Parametri dei semafori (sempre verde per semplificare)
    traffic_lights = TrafficLightParams()

    # Parametri della simulazione SUMO
    sim = SumoParams(
        sim_step=0.1,
        render=False,
        restart_instance=True,
    )

    # Parametri dell'ambiente
    env = EnvParams(
        horizon=HORIZON,
        additional_params={
            'target_velocity': TARGET_VELOCITY,
            'max_accel': MAX_ACCEL,
            'max_decel': MAX_DECEL,
            'sort_vehicles': False
        },
    )

    
    #initial configuration for inflow
    initial = InitialConfig(spacing="random")

    #network configuration
    net_params = NetParams(
        inflows=inflow,
        additional_params={
            'speed_limit': 30,
        }
    )
    
    # Parametri di Flow
    flow_params = dict(
        # Nome dell'esperimento
        exp_tag="DDPG_Intersection_new_env",
        # Nome dell'ambiente Flow
        env_name= VariantNormalizedMultiAgentAccelPOEnv if rl_flag else TestEnv, 
        # Classe della rete
        network=Intersection,
        # Parametri del simulatore
        simulator='traci',
        sim=sim,
        # Parametri dell'ambiente
        env=env,
        # Parametri della rete
        net=net_params,
        # Veicoli nella rete
        veh=vehicles,
        # Configurazione iniziale
        initial=initial,
        # Parametri dei semafori
        traffic_lights=traffic_lights
    )

    return flow_params

