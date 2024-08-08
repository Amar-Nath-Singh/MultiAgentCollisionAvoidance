import os
import glob

if input('Delete saved Log(Y/n):').lower()[0] == 'y':
    files = glob.glob('saved_npy/*')
    for f in files:
        print('## Removing', f)
        os.remove(f)

import ray
from gym.spaces import Discrete, Box
from ray.rllib.algorithms.dqn import DQN
from ray.rllib.algorithms.ppo import PPO
from multi_agent_env import Multi_Agent_Env
import numpy as np
from utils import *
from kcs_ode import KCS_ode
from scipy.integrate import solve_ivp


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
    
params = {}
num_agents = 5
Fmag = np.inf
action_space = Discrete(5)
obs_space = Box(low = np.array([-np.inf, -np.pi, -np.inf, -1, -Fmag,-Fmag,-Fmag,-Fmag],dtype=np.float64), 
                high = np.array([np.inf, np.pi, np.inf, 1, Fmag,Fmag,Fmag,Fmag],dtype=np.float64))
min_goal = 15.0
max_goal = 20.0


# u,v,r,x,y,yaw,delta_c, nprop, x_init, y_init, x_goal, y_goal
agent_init = np.zeros((12, num_agents), dtype=np.float64)
agent_init[0] = agent_init[0] + 1 # u = 1
agent_init[3:5] = np.random.uniform(max_goal, 100 - max_goal, (2,num_agents))
agent_init[5] = np.random.uniform(-np.pi, np.pi, num_agents)
agent_init[7] = agent_init[7] + 115.5/60
agent_init[8:10] = agent_init[3:5].copy()
theta = tan_inv(agent_init[3], agent_init[4])
agent_init[10:12] = agent_init[8:10] + np.random.uniform(min_goal, max_goal, num_agents) * np.array([np.cos(theta), np.sin(theta)])


params['grid_size'] = 120
params['action_space'] = action_space
params['observation_space'] = obs_space
params['agents'] = agent_init.T.copy()
params['min_goal'] = min_goal
params['max_goal'] = max_goal
params['agent'] = action_space

params['R0'] = 30.0
params['coa'] = 0.5
params['collision_radius'] = 1.0 + params['coa']
params['manuvering_radi'] = 3.0
params['update_fxn_dict'] = {_id: update_fxn for _id in list(range(params['agents'].shape[0]))}
params['max_step'] = 150

# params['render_node'] = rclpy.create_node('MultiAgentEnv')
def policy_mapping_fn(agent_id, episode, worker, **kwargs):
    return f"policy_1"

exploration_config =  {'type': 'EpsilonGreedy', 'initial_epsilon': 1.0, 'final_epsilon': 0.05, 'epsilon_timesteps': 25000}

lr = 0.0001

# Configure the environment and agents
config = {
    "env": Multi_Agent_Env,
    "env_config": params,
    "multiagent": {
        "policies": {
            "policy_1": (None, obs_space, action_space, {}),
        },
        "policies_to_train": ['policy_1'],
        "policy_mapping_fn": policy_mapping_fn,
    },

    "num_workers": 5,
    "num_gpus": 1,

    'model':{
            "fcnet_hiddens": [128, 128]
        },

        "replay_buffer_config" : {
                "_enable_replay_buffer_api": True,
                "type": "MultiAgentPrioritizedReplayBuffer",
                "capacity": 100000,
                "prioritized_replay_alpha": 0.6,
                "prioritized_replay_beta": 0.4,
                "prioritized_replay_eps": 1e-6,
                "replay_sequence_length": 1,
                },

        "exploration_config": exploration_config,

        "lr": lr,

        "dueling": False,

        "tau": 0.02,
        'gamma':0.98,

}

trainer = DQN(config=config)
print('INIT')
for n in range(100):
    print()
    result = trainer.train()
    print(f"{n + 1}: {result['episode_reward_mean']}")

ray.shutdown()
