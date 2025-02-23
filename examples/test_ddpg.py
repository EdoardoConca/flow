from ray import tune
from ray.tune.registry import register_env
import gym
import numpy as np

import os
import ray

# 🔥 Disabilita l'uso di pyarrow per la serializzazione
os.environ["RAY_DISABLE_PYARROW"] = "1"
os.environ["RLLIB_DISABLE_COMPRESSION"] = "1"

def env_creator(config):
    env = gym.make("Pendulum-v0")

    class NumpyArrayWrapper(gym.ObservationWrapper):
        def observation(self, obs):
            print(f"Osservazione prima della conversione: {obs}, tipo: {type(obs)}")
            obs = np.array(obs, dtype=np.float32)
            print(f"Osservazione dopo la conversione: {obs}, tipo: {type(obs)}, shape: {obs.shape}")
            return obs

    return NumpyArrayWrapper(env)

# Registra l'ambiente Pendulum
register_env("Pendulum-v0", env_creator)

# Configurazione per DDPG
n_cpus = 1  # Numero di CPU da utilizzare
horizon = 200  # Orizzonte temporale (ad esempio, 200 passi per episodio)
n_rollouts = 1  # Numero di rollout per batch di training

config = {
        "env": "Pendulum-v0",  # Ambiente continuo
        "num_workers": n_cpus,
        "train_batch_size": horizon * n_rollouts,  # Dimensione del batch di training
        "gamma": 0.99,  # Fattore di sconto
        "actor_hiddens": [400, 300],  # Rete neurale per l'attore
        "critic_hiddens": [400, 300],  # Rete neurale per il critico
        "actor_lr": 1e-4,  # Learning rate per l'attore
        "critic_lr": 1e-3,  # Learning rate per il critico
        "tau": 0.001,  # Tasso di aggiornamento soft per i target networks
        "buffer_size": 1000000,  # Dimensione del replay buffer
        "exploration_noise_type": "ou",  # Tipo di rumore di esplorazione
        "exploration_ou_theta": 0.15,  # Parametro theta per il rumore di Ornstein-Uhlenbeck
        "exploration_ou_sigma": 0.2,  # Parametro sigma per il rumore di Ornstein-Uhlenbeck
        "exploration_ou_noise_scale": 0.1,  # Scala del rumore di Ornstein-Uhlenbeck
        "horizon": horizon,  # Orizzonte temporale
        "compress_observations": False
}

# Esegui il training con DDPG
tune.run("DDPG", config=config)
