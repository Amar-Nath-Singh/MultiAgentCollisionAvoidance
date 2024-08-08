from ray.rllib.env.multi_agent_env import MultiAgentEnv
import numpy as np
from utils import *
import random

class Multi_Agent_Env(MultiAgentEnv):
    def __init__(self, params: dict):
        super().__init__()
        self.params = params.copy()
        self.observation_space = self.params['observation_space']
        self.action_space = self.params['action_space']

        self.terminateds = set()
        self.truncateds = set()

        # Agent state [u,v,r,x,y,yaw, delta_c, nprop, x_init, y_init, x_goal, y_goal]
        self.agents : np.ndarray = self.params['agents']  # Numpy  Array of shape Num_Agent x State
        self.agent_update_fxn = self.params['update_fxn_dict'] # dict of update_fxn for all the agents in numpy array

        self._agent_ids = list(range(self.agents.shape[0]))
        self.num_agents = len(self._agent_ids)
        self.resetted = False

        self.count_step = 0
        self.save_list = np.zeros((self.params['max_step'] + 1, self.agents.shape[0], self.agents.shape[1]), dtype=np.float64)
        self.episode_step = 0

        self.spacing_dist = self.params['spacing_dist']

        self.mode = self.params['mode']

    def reset(self, *, seed = None, options = None):
        super().reset(seed=seed)
        self.resetted = True
        if random.random() < 0.1:
            np.save(f'saved_npy/EP_{self.episode_step}',self.save_list)
        # print(f'## Episode[{self.episode_step}] Saved ##')
        self.episode_step += 1
        self.save_list = self.save_list * 0

        self.count_step = 0
        self.terminateds = set()
        self.truncateds = set()
        observation, info = {}, {}
        i = 0

        visited_points = []

        agent_ids = list(self._agent_ids)
        while i < len(agent_ids):
            
            x_init, y_init = np.random.uniform(0, self.params['grid_size'], 2)
            dist = np.random.uniform(self.params['min_goal'], self.params['max_goal'])
            direction = np.random.uniform(-np.pi, np.pi)
            x_goal, y_goal = x_init + (np.cos(direction) * dist), y_init + (np.sin(direction) * dist)
            yaw = np.random.uniform(-np.pi, np.pi)

            invalid = False
            for pt in visited_points:
                if distance(x_init - pt[0], y_init - pt[1]) < self.spacing_dist: # or distance(x_goal - pt[0], y_goal- pt[1]) < self.spacing_dist:
                    invalid = True
                    break

            if invalid:
                continue

            visited_points.append((x_init, y_init))

            agent_id = agent_ids[i]
            self.agents[agent_id] = np.array([1.0, 0, 0, x_init, y_init, yaw, 0, self.agents[agent_id][7],
                                               x_init, y_init, x_goal, y_goal], dtype=np.float64)
            observation[agent_id],_,_,_,info[agent_id]  = self.agent_step(agent_id)

            # print(f"## Spwaned [{i}]")
            i += 1

        return observation, info
    
    def agent_update(self, agent_id, action):
        self.agents[agent_id] = self.agent_update_fxn[agent_id](self.agents[agent_id], action)

    def agent_step(self, agent_id):

        agent_idx_mask = np.ones(len(self._agent_ids), dtype=bool)
        agent_idx_mask[agent_id] = False
        agent = self.agents[agent_id].T
        other_agents = self.agents[agent_idx_mask].T

        agent_state = agent[:6]
        goal = agent[8:]
        obstacles = other_agents[:6].T
        goal_obs, goal_reward, goal_done, _ = goal_step(state = agent_state, goal=goal, min_goal=self.params['min_goal'], max_goal=self.params['max_goal'], coa=self.params['coa'])


        obstacle_obs, obstacle_reward, collision, _ = APF_step(state=agent_state, obstacles=obstacles, 
                                                               max_range=self.params['R0'],
                                                               col_radi=self.params['collision_radius'])
        truncated = False
        if self.mode == 'pf':
            observation = goal_obs
            reward = goal_reward
            terminated = goal_done
            info = None
        elif self.mode == 'sc':
            observation = np.hstack([goal_obs, obstacle_obs[:2]], dtype=np.float64)
            reward = goal_reward
            if collision:
                reward = -200.0
            terminated = goal_done or collision
            info = None
        else:
            observation = np.hstack([goal_obs, obstacle_obs], dtype=np.float64)
            reward = goal_reward + (self.params['rew_scale'] * obstacle_reward)
            if collision:
                reward = -200.0
            terminated = goal_done or collision
            info = None

        return observation, reward, terminated, truncated, info
    
    def step(self, actions):
        observation_dict, reward_dict, terminated_dict, truncated_dict, info_dict = {}, {}, {}, {}, {}
        for agent_id, action in actions.items():
            self.agent_update(agent_id, action)

        for agent_id, action in actions.items():
            observation, reward, terminated, truncated, info = self.agent_step(agent_id)
            if terminated:
                self.terminateds.add(agent_id)
            if truncated:
                self.truncateds.add(agent_id)

            observation_dict[agent_id], reward_dict[agent_id], terminated_dict[agent_id], truncated_dict[agent_id], _ = observation, reward, terminated, truncated, info
            
        terminated_dict["__all__"] = len(
            self.terminateds) == self.num_agents
        truncated_dict["__all__"] = self.count_step >= self.params['max_step']
        self.save_list[self.count_step] = self.agents.copy()
        self.count_step +=1
        
        return observation_dict, reward_dict, terminated_dict, truncated_dict, info_dict
    
    def render(self) -> None:
        super().render()







            

