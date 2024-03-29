from envs.level0.tmps_env import env_level0
import numpy as np


class explore_wrapper(env_level0):

    def __init__(self, configs):
        super().__init__(configs)
        # number of agent
        self.n = 4
        # number of enermy
        self.en = 4

    def reset(self, **kwargs):

        env_level0.reset(self, **kwargs)

        return self.get_observation()
        
    def step(self, actions): # assumes 'actions' is 4(agent) X 2(accel,alpha) matrix

        action_tuple = [] # initialization

        for row in actions:
            # CAUTION! : self.action[4] == 0:  # EXPLORE MODE
            action_tuple.append((np.array([row[0]]), np.array([row[1]]), 0, 7, 0))

        _, _, agent_done, agent_info = env_level0.step(self, action_tuple)

        agent_obs = self.get_observation()
        agent_reward = self.compute_reward(agent_obs)

        return agent_obs, agent_reward, agent_done, agent_info
    
    def compute_reward(self, obs):
        # print(len(self.agents[0].list_detected_enemy))
        # print(self.agents[0].position)
        # Give large bonus if all enemies are detected
        
        if len(self.agents[0].list_detected_enemy) >= 1: # TODO 좀 더 정교한 reward를 위해서는 '발견한 agent'에게 reward 줘야함 
            detection_bonus = np.array([1, 1, 1 ,1])*len(self.agents[0].list_detected_enemy) # check 이렇게 주면, 한 명 발견하고 만족해서 더 explore 안할수도 있음. 그래서 1로 줌.
            if len(self.agents[0].list_detected_enemy) == self.en:
                detection_bonus = np.array([1000, 1000, 1000 ,1000])
                print('All enemies DETECTED!!')
        else :
            detection_bonus = np.array([0, 0, 0 ,0])

        rel_pos = obs[:, -8:] # get rel pos array
        rel_pos_assigned = np.array([rel_pos[0, [0, 1]],  # 1row - (1, 2)
                                     rel_pos[1, [2, 3]],  # 2row - (3, 4)
                                     rel_pos[2, [4, 5]],  # 3row - (5, 6)
                                     rel_pos[3, [6, 7]]]) # 4row - (7, 8)
        
        # negative L2 distance
        l2_norms_neg = -np.linalg.norm(rel_pos_assigned, axis=1)

        # toal reward
        agent_reward = detection_bonus + 0.0001*l2_norms_neg # 0.01 -> 0.01 * 0.001. 이렇게 안 하면 리워드가 -1300 이렇게 나오더라. 02/18 DHO -> 0.0001 pjhae

        return agent_reward


    def get_observation(self):

        # agent global pos
        objects_pos = np.zeros((len(self.agents) + len(self.enemies), 2), dtype=np.float32)
        for i, agent in enumerate(self.agents):
            objects_pos[i] = (agent.position[0], agent.position[1])
        for i, enemy in enumerate(self.enemies):
            objects_pos[len(self.agents) + i] = (enemy.position[0], enemy.position[1])

        # object yaw, (sin, cos)
        yaws = np.array([agent.current_orientation for agent in self.agents])
        sinusoidal_yaw_matrix = np.column_stack((np.cos(yaws), np.sin(yaws)))

        # enemy-agent relative pos
        allies = objects_pos[:4, :]
        enemies = objects_pos[4:, :]
        rel_pos_matrix = np.empty((0, 8))
        for ally in allies:
            relative_positions = enemies - ally  # rel pos
            relative_vector = relative_positions.flatten()  
            rel_pos_matrix = np.vstack([rel_pos_matrix, relative_vector])  # concat

        # concat all : 4X12 행렬, 12 = 2(global x,y) + 2(cosyaw,sinyaw) + 2[rel x,y]*4
        concatenated_obs = np.hstack([allies, sinusoidal_yaw_matrix, rel_pos_matrix])

        return concatenated_obs