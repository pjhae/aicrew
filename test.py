import os
import sys

import torch
import torch.nn.functional as F
from envs.level0.explore_wrapper import explore_wrapper
from arguments import parse_args

def get_trainers(env, arglist):

    model_epi_number = 2849900

    """ load the model """
    actors_tar = [torch.load(arglist.old_model_name+'aicrew_{}/'.format(model_epi_number)+'a_c_{}.pt'.format(agent_idx), map_location=arglist.device) \
        for agent_idx in range(env.n)]

    return actors_tar


def test(arglist):
    """ 
    This func is used for testing the model
    """
    env_config = {
        "num_agents": 4,
        "obs_box_size": 50,
        "init_pos": ((55., 30.), (75., 30.), (95., 25.), (105., 30.)),
        # "init_pos": ((140., 140.), (160., 140.), (140., 160.), (160., 160.)),
        # "init_pos": ((88, 69), (190, 120), (67, 220), (195, 220)),
        "dynamic_delta_t": 1.1
    }
    # 참고 : 'enemy_init_pos': ((88, 69), (190, 120), (67, 220), (195, 220))

    """ init the env """
    env = explore_wrapper(env_config)

    episode_step = 0

    """ init the agents """
    actors = get_trainers(env, arglist)

    """ interact with the env """
    reset_arg = {'episode': 0}
    obs_n = env.reset(**reset_arg)

    while True:

        # update the episode step number
        episode_step += 1

        # get action
        action_n = [agent(torch.from_numpy(obs).to(arglist.device, torch.float), eval=True).detach().cpu().numpy() \
            for agent, obs in zip(actors, obs_n)]

        # interact with env
        obs_n, rew_n, done_n, info_n = env.step(action_n)

        # update the flag
        done = done_n
        terminal = (episode_step >= arglist.per_episode_max_len)

        # reset the env
        if done or terminal: 
            episode_step = 0
            obs_n = env.reset(**reset_arg)

        # render the env
        #print(rew_n)
        env.render()

if __name__ == '__main__':

    arglist = parse_args()
    test(arglist)
