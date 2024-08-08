from kcs_ode import KCS_ode
from scipy.integrate import solve_ivp
import numpy as np
from utils import *
import sys

np.random.seed(int(sys.argv[1]))

def update_fxn(state, delta_c):
    tspan = (0, 0.3)
    yinit = state[:7]
    sol = solve_ivp(
        lambda t, v: KCS_ode(t, v, delta_c),
        tspan, yinit, t_eval=tspan, dense_output=True)
    state[:7] = np.array(sol.y, dtype=np.float64).T[-1]
    state[5] = ssa(state[5])
    return state

def APF(agent, obstacles, K_Att, K_rep, K_vel):
    agent_state = agent[:6]
    x_init, y_init, Xg, Yg = goal = agent[-4:]
    obstacles = obstacles[:6].T
    u, v, r, X, Y, YAW = agent_state

    goal_obs, goal_reward, goal_done, _ = goal_step(agent_state, goal, 25, 50, 0.5)

    obstacle_obs, obstacle_reward, collision, _ = APF_step(state=agent_state, obstacles=obstacles, 
                                                            max_range = 30, col_radi = 1.5)
    
    if collision:
        print('collision')

    if goal_done:
        print('Goal')

    FxG, FyG = BCS(Xg, Yg, X, Y, YAW)

    FxO,FyO,FxV,FyV = obstacle_obs

    Fx = K_Att * FxG + K_rep * FxO + K_vel * FxV
    Fy = K_Att * FyG + K_rep * FyO + K_vel * FyV

    desired_yaw_local = np.arctan2(Fy, Fx)

    delta_c = 2 * desired_yaw_local - 4 * r

    return delta_c, goal_done or collision

if __name__ == '__main__':
    num_agents = 10
    agents = np.zeros((num_agents, 12), dtype=np.float64)

    i = 0
    visited_points = []
    while i < num_agents:
        
        x_init, y_init = np.random.uniform(0, 50, 2)
        dist = np.random.uniform(25, 50)
        direction = np.random.uniform(-np.pi, np.pi)
        x_goal, y_goal = x_init + (np.cos(direction) * dist), y_init + (np.sin(direction) * dist)
        yaw = np.random.uniform(-np.pi, np.pi)

        invalid = False
        for pt in visited_points:
            if distance(x_init - pt[0], y_init - pt[1]) < 10: # or distance(x_goal - pt[0], y_goal- pt[1]) < self.spacing_dist:
                invalid = True
                break

        if invalid:
            continue

        print(f'## Spawned [{i}]')

        visited_points.append((x_init, y_init))

        agents[i] = np.array([1.0, 0, 0, x_init, y_init, yaw, 0, 115.5/60,
                                            x_init, y_init, x_goal, y_goal], dtype=np.float64)
        i += 1

    save_list = []
    terminateds = set()
    while (not terminateds.__len__() == num_agents) and len(save_list) < 200:
        actions = {}
        for agent_id in range(num_agents):
            if agent_id in terminateds:
                continue

            agent_idx_mask = np.ones(num_agents, dtype=bool)
            agent_idx_mask[agent_id] = False
            agent = agents[agent_id].T
            other_agents = agents[agent_idx_mask].T
            delta_c, terminated = APF(agent=agent, obstacles=other_agents, K_Att=0.05, K_rep = -10.0, K_vel = 0.005)

            delta_c = np.clip(delta_c, -np.radians(35), np.radians(35))
            actions[agent_id] = delta_c
            if terminated:
                terminateds.add(agent_id)

        agent_id = 0        
        for (agent_id, action) in actions.items():
            if agent_id in terminateds:
                continue
            # print(agent_id, action)
            agents[agent_id] = update_fxn(state=agents[agent_id], delta_c = action)
        
        save_list.append(agents.copy())

    np.save('saved_npy/EP_0', np.array(save_list))
    print('Saved..')

    