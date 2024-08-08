import os
import glob

if input('Delete saved Log(Y/n):').lower()[0] == 'y':
    files = glob.glob('saved_npy/*')
    for f in files:
        print('## Removing', f)
        os.remove(f)

import ray
import ray.tune as tune
from ray import air
from ray.rllib.algorithms.appo import APPOConfig

from gym.spaces import Discrete, Box
from multi_agent_env import Multi_Agent_Env
import numpy as np
from utils import *
from kcs_ode import KCS_ode
from scipy.integrate import solve_ivp


def env_creator(env_config):
        return Multi_Agent_Env(**env_config)

tune.registry.register_env('multi_agent_env', env_creator)

action_map = {
    0: np.radians(-35),
    1: np.radians(-20),
    2: np.radians(0),
    3: np.radians(20),
    4: np.radians(35),
}

def update_fxn(state, action):
    tspan = (0, 0.3)
    yinit = state[:7]
    delta_c = action_map[action]
    sol = solve_ivp(
        lambda t, v: KCS_ode(t, v, delta_c),
        tspan, yinit, t_eval=tspan, dense_output=True)
    state[:7] = np.array(sol.y, dtype=np.float64).T[-1]
    state[5] = ssa(state[5])
    return state
    

num_agents = 10

params = {}
params['mode'] = 'sc' # 'Path Following': 'pf', 'Static Obstalces': 'sc', "Dynamic Obstalces": 'dc'
Fmag = np.inf
if params['mode'] == 'pf':
     obs_space = Box(low = np.array([-np.inf, -np.pi, -np.inf, -1],dtype=np.float64), 
                high = np.array([np.inf, np.pi, np.inf, 1],dtype=np.float64))
elif params['mode'] == 'sc':
     obs_space = Box(low = np.array([-np.inf, -np.pi, -np.inf, -1, -Fmag,-Fmag],dtype=np.float64), 
                high = np.array([np.inf, np.pi, np.inf, 1, Fmag, Fmag],dtype=np.float64))
else:
    params['mode'] = 'dc'
    obs_space = Box(low = np.array([-np.inf, -np.pi, -np.inf, -1, -Fmag,-Fmag,-Fmag,-Fmag],dtype=np.float64), 
                high = np.array([np.inf, np.pi, np.inf, 1, Fmag,Fmag,Fmag,Fmag],dtype=np.float64))
action_space = Discrete(5)
min_goal = 18.0
max_goal = 25.0


# u,v,r,x,y,yaw,delta_c, nprop, x_init, y_init, x_goal, y_goal
agent_init = np.zeros((12, num_agents), dtype=np.float64)
agent_init[0] = agent_init[0] + 1 # u = 1
agent_init[3:5] = np.random.uniform(max_goal, 100 - max_goal, (2,num_agents))
agent_init[5] = np.random.uniform(-np.pi, np.pi, num_agents)
agent_init[7] = agent_init[7] + 115.5/60
agent_init[8:10] = agent_init[3:5].copy()
theta = tan_inv(agent_init[3], agent_init[4])
agent_init[10:12] = agent_init[8:10] + np.random.uniform(min_goal, max_goal, num_agents) * np.array([np.cos(theta), np.sin(theta)])


params['grid_size'] = 40.0
params['spacing_dist'] = 10.0

params['action_space'] = action_space
params['observation_space'] = obs_space
params['agents'] = agent_init.T.copy()
params['min_goal'] = min_goal
params['max_goal'] = max_goal
params['agent'] = action_space

params['R0'] = 30.0
params['coa'] = 0.5
params['collision_radius'] = 1.5 + params['coa']
params['manuvering_radi'] = 3.0
params['update_fxn_dict'] = {_id: update_fxn for _id in list(range(params['agents'].shape[0]))}
params['max_step'] = 150

params['rew_scale'] = 0.1

# params['render_node'] = rclpy.create_node('MultiAgentEnv')
def policy_mapping_fn(agent_id, episode, worker, **kwargs):
    return f"policy_1"

def exp_schedule(initial_value: float, final_value: float, total_timesteps: int):

    def func(progress_remaining: float) -> float:
        if not (0.0 <= progress_remaining <= 1.0):
            raise ValueError("Progress remaining should be between 0.0 and 1.0")

        decay_factor = (final_value / initial_value) ** (1 / total_timesteps)
        current_learning_rate = initial_value * (decay_factor ** ((1 - progress_remaining) * total_timesteps))

        return current_learning_rate
    return func

training_iteration = 2000
config = APPOConfig()
config.lr = 0.0005
config.lr_schedule = None
config.train_batch_size = 128
config.tau = 0.02
config.gamma = 0.98
config.model = {
    "fcnet_hiddens": [256, 256],
    "fcnet_activation": "tanh",
    # Whether to wrap the model with an LSTM.
    "use_lstm": True,
    # Max seq len for training the LSTM, defaults to 20.
    "max_seq_len": 10,
    # Size of the LSTM cell.
    "lstm_cell_size": 64,
    # Whether to feed a_{t-1} to LSTM (one-hot encoded if discrete).
    "lstm_use_prev_action": tune.grid_search([False, True]),
    # Whether to feed r_{t-1} to LSTM.
    "lstm_use_prev_reward": tune.grid_search([False, True]),
}

config.policy_mapping_fn = policy_mapping_fn
config.policies_to_train = ['policy_1']
config.policies = {
            "policy_1": (None, obs_space, action_space, {}),
        }

config.env = Multi_Agent_Env
config.env_config = params
config.num_rollout_workers = 10
config.num_gpus = 1
config.simple_optimizer = False

config.batch_mode = 'truncate_episodes'

config.train_batch_size = 512

tuner = tune.Tuner(
            "APPO",
            run_config=air.RunConfig(stop={"training_iteration":training_iteration}),
            param_space=config.to_dict()
        )
tuner.fit()


ray.shutdown()
