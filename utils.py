import numpy as np

def tan_inv(X,Y):
    return np.arctan2(Y, X)

def distance(X, Y):
    return np.sqrt(np.square(X) + np.square(Y))

def ssa(angle):
    return (angle + np.pi)%(2*np.pi) - np.pi

def GCS(x1, y1, x, y, theta):
    xr, yr = x1 - x, y1 - y
    xn = xr * np.cos(theta) -yr * np.sin(theta)
    yn = xr * np.sin(theta) + yr * np.cos(theta)
    return xn, yn

def BCS(x1, y1, x, y, theta):
    xr, yr = x1 - x, y1 - y
    xn = xr * np.cos(theta) + yr * np.sin(theta)
    yn = -xr * np.sin(theta) + yr * np.cos(theta)
    return xn, yn

def norm(X, Y):
    return np.sqrt(np.square(X) + np.square(Y))

def goal_step(state, goal, min_goal, max_goal, coa):
    u, v, r, x, y, psi = state
    x_init, y_init, x_goal, y_goal = goal

    D = distance(x_goal-x, y_goal-y)
     # CROSS TRACK ERROR
    vec1 = np.array([x_goal - x_init, y_goal - y_init])
    vec2 = np.array([x_goal - x, y_goal - y])

    vec1_hat = vec1 / np.linalg.norm(vec1)
    cross_track_error = float(np.cross(vec2, vec1_hat))
    x_dot = u * np.cos(psi) - v * np.sin(psi)
    y_dot = u * np.sin(psi) + v * np.cos(psi)

    Uvec = np.array([x_dot, y_dot])
    Uvec_hat = Uvec / np.linalg.norm(Uvec)
    vec2_hat = vec2 / np.linalg.norm(vec2)

    course_angle = np.arctan2(Uvec[1], Uvec[0])
    psi_vec2 = np.arctan2(vec2[1], vec2[0])

    course_angle_err = course_angle - psi_vec2
    course_angle_err = (course_angle_err + np.pi) % (2 * np.pi) - np.pi

    # y_min + ((x - x_min) / (x_max - x_min)) * (y_max - y_min)

    Dg_inp = (D / max_goal) * 18
    R1 = 2 * np.exp(-0.08 * cross_track_error ** 2) - 1.0
    R2 = 1.3 * np.exp(- 10.0 * (abs(course_angle_err))) - 0.3    
    R3 = -0.25 * Dg_inp
    reward = R1 + R2 + R3
    done_goal = False

    dir_Vel = np.dot(vec2_hat, Uvec_hat)
    dir_path = np.dot(vec1_hat, vec2_hat)
    angle_btw23 = np.arccos(dir_Vel)
    angle_btw12 = np.arccos(dir_path)

    if D < coa:
        done_goal = True
        reward = 20.0
        # print('Reached!', D, coa, [cross_track_error, course_angle_err, Dg_inp, r], reward)
    elif angle_btw12 > np.pi / 2 and angle_btw23 > np.pi / 2:
        # print(f"Destination missed!")
        done_goal = True

    return np.array([cross_track_error, course_angle_err, Dg_inp, r], dtype=np.float64), reward, done_goal, {}


def APF_step(state, obstacles, max_range, col_radi):
    info = {}
    done = False
    reward = 0.0
    observation = np.zeros(4, dtype=np.float64)
    u,v,r,x,y,psi = state
    u_o,v_o,r_o,x_o,y_o,psi_o = obstacles.T

    u,v = GCS(u,v,0,0,psi)
    Vox, Voy = GCS(u_o, v_o, 0,0, psi_o)

    k1 = 1
    k2 = 1

    Xr, Yr = BCS(x_o,y_o,x,y,psi)
    Vrx,Vry = BCS(Vox,Voy, u,v,psi)

    D = norm(Xr, Yr)
    range_mask = D < max_range
    D = D[range_mask]
    Xr = Xr[range_mask]
    Yr = Yr[range_mask]
    Vrx = Vrx[range_mask]
    Vry = Vry[range_mask]

    if not D.shape[0] > 0:
        return observation, reward, done, info
    
    idx_min = np.argmin(D)
    min_D = D[idx_min]
    theta_l = ssa(np.arctan2(Yr, Xr) - np.pi)
    K = (col_radi**2)/((1/col_radi) - (1/max_range))
    Fp= K * ((1/D) - (1/max_range)) / np.square(D)
    Fpx = np.sum(Fp * np.cos(theta_l))
    Fpy = np.sum(Fp * np.sin(theta_l))

    Fvx = max_range * np.sum(Vrx/D)
    Fvy = max_range * np.sum(Vry/D)
    
    chi_rel = np.arctan2(Vry,Vrx)
    r_theta = np.arctan2(Yr,Xr)
    gamma = ssa(chi_rel - r_theta - np.pi)
    DCPA = np.abs(np.sin(gamma))*D
    TCPA = np.cos(gamma)*r / norm(Vrx,Vry)
    CRI = np.exp(-(k1*DCPA + k2*TCPA))
    CRI[TCPA<=0] = 0
    CRI[D < col_radi] = 1.0
    
    observation = np.array([Fpx,Fpy,Fvx,Fvy], dtype=np.float64)
    reward = -np.max(CRI)
    if min_D < col_radi:
        done = True
        # print('Collision!',CRI ,reward)

    return observation, reward, done, info


def r_mat(psi):

    R = np.matrix([[np.cos(psi), -np.sin(psi)], [np.sin(psi), np.cos(psi)]])

    return R

def agent_step(self, agent_id):
        agent_idx_mask = np.ones(len(self._agent_ids), dtype=bool)
        agent_idx_mask[agent_id] = False
        agent = self.agents[agent_id].T
        other_agents = self.agents[agent_idx_mask].T

        u, v, r, x, y, yaw, delta_c, nprop, x_init, y_init, x_goal, y_goal = agent
        u_obs, v_obs, _, x_obs, y_obs, yaw_obs, _,_,_,_, _, _ = other_agents

        R0 = self.params['R0']
        coli_radi = self.params['collision_radius']
        min_goal = self.params['min_goal']
        max_goal = self.params['max_goal']
        coa = self.params['coa']

        # Global Vx, Vy of agent
        U_g, V_g = GCS(u,v,0,0,yaw)

        # Global Vx, Vy of other agents
        U_obs_g, V_obs_g = GCS(u_obs, v_obs, 0, 0, yaw_obs)

        # Relative Pose X, Y in agent BCS
        X, Y = BCS(x_obs, y_obs, x, y, yaw) 

        # Relative Velocity Vx, Yy in agent BCS
        Vx, Vy = BCS(U_obs_g, V_obs_g, U_g, V_g, yaw)

        # Relative pose in Polar coordinates
        R = distance(X, Y)
        theta = tan_inv(X, Y)

        K = np.square(coli_radi) / ((1/coli_radi) - (1/R0))
        range_mask = R < R0
        R = R[range_mask]
        Fx = 0.0
        Fy = 0.0
        Fvx = 0.0
        Fvy = 0.0
        
        if R.shape[0] > 0:
            theta = theta[range_mask]
            Vx = Vx[range_mask]
            Vy = Vy[range_mask]

            # List of Magnitude of Force
            Fmag_i = ((1/R) - (1/R0)) / np.square(R)

            # List of Magnitude of Force in X wrt Agent BCS
            Fx_i = Fmag_i * np.cos(theta)
            # List of Magnitude of Force in Y wrt Agent BCS
            Fy_i = Fmag_i * np.sin(theta)

            Fx = K * np.sum(Fx_i)
            Fy = K * np.sum(Fy_i)

            Fvx = R0 * np.sum(Vx / R)
            Fvy = R0 * np.sum(Vy / R)

        # CROSS TRACK ERROR
        vec1 = np.array([x_goal - x_init, y_goal - y_init])
        vec2 = np.array([x_goal - x, y_goal - y])
        vec1_hat = vec1 / np.linalg.norm(vec1)
        cross_track_error = np.cross(vec2, vec1_hat)
        x_dot = u * np.cos(yaw) - v * np.sin(yaw)
        y_dot = u * np.sin(yaw) + v * np.cos(yaw)

        Uvec = np.array([x_dot, y_dot])
        Uvec_hat = Uvec / np.linalg.norm(Uvec)
        vec2_hat = vec2 / np.linalg.norm(vec2)

        course_angle = np.arctan2(Uvec[1], Uvec[0])
        psi_vec2 = np.arctan2(vec2[1], vec2[0])

        course_angle_err = course_angle - psi_vec2
        course_angle_err = (course_angle_err + np.pi) % (2 * np.pi) - np.pi
        
        goal_distance = distance(x_goal-x, y_goal - y)
        # Dg_inp = 8 + (((goal_distance - min_goal)) * (18 - 8)/(max_goal - min_goal))
        Dg_inp = goal_distance
        R1 = 2 * np.exp(-0.08 * cross_track_error ** 2) - 1.0
        R2 = 1.3 * np.exp(- 10.0 * (abs(course_angle_err))) - 0.3    
        R3 = -0.25 * Dg_inp
        reward_goal = R1 + R2 + R3
        angle_btw23 = np.arccos(np.dot(vec2_hat, Uvec_hat))
        angle_btw12 = np.arccos(np.dot(vec1_hat, vec2_hat))

        truncated = False
        terminated = False
        info = None

        observation = np.array([cross_track_error, course_angle_err, Dg_inp, r,
                                      Fx, Fy, Fvx, Fvy])
        reward = reward_goal
        if goal_distance < coa:
            terminated = True
            reward = 20.0
            # print('Reached !')
        elif angle_btw12 > np.pi / 2 and angle_btw23 > np.pi / 2:
            terminated = True
            # print('Passed Goal !')
        else:
            terminated = False

        if (R < coli_radi).any() == True:
            terminated = True
            reward = -100.0
            # print('Collision !')

        return observation, reward, terminated, truncated, info
