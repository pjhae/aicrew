
from envs.level0.tmps_env import env_level0
import envs.level0
import pygame
import array
import numpy as np
from numpy import array

env_config = {
    "num_agents": 4,
    "obs_box_size": 50,
    #  "init_pos": ((55., 30.), (75., 30.), (95., 25.), (105., 30.)),
    # "init_pos": ((55., 130.), (175., 260.), (195., 25.), (105., 103.)),
    "init_pos" : ((20., 50.), (191, 121), (71, 221), (196, 221)),
    "dynamic_delta_t": 1.1
}

if __name__ == '__main__':
    env = env_level0(env_config)
    agents = [f'agent_{i}' for i in range(env_config['num_agents'])]

    # Start an episode!
    # Each observation from the environment contains a list of observaitons for each agent.
    # In this case there's only one agent so the list will be of length one.

    # obs_list = env.reset()


    reset_arg = {
        'episode': 0
    }
    # episode 2000
    for i in range(2000):
        done = False
        steps = 0
        reset_arg['episode'] = i
        env.reset(**reset_arg)

        print("enemies: ", len(env.agents[0].list_detected_enemy))

        while not done and steps < 1000000:
            steps += 1
            # print(steps)
            env.render()  # OPTIONAL: render the whole scene + birds eye view

            _act = env.action_space.sample()
            # print('action_space_sample ', _act)
            
            _obs = env.observation_space.sample()
            # print('observation_space_sample ', _obs)
            agent_obs, agent_rew, agent_done, agent_info = env.step(((array([0.71834777]), array([-0.39035561]), 0, 7, 1), (array([-1.58439287]), array([-0.43161788]), 0, 7, 1), (array([0.76046334]), array([-0.33415608]), 0, 7, 1), (array([-2.49030743]), array([0.0911108]), 0, 7, 1)))
            # print(agent_rew, agent_done, agent_info)
            agents = agent_info['agents']
            # if agents[1].num_detected_enemy() > 0:
            #     print("closest", agents[1].closest_detected_enemy_id, agents[1].closest_detected_enemy_dist, "far", agents[1].most_far_placed_enemy_dist)

            done = agent_done


        print("episode:", i, "steps:", steps)
print("close")

