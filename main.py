# Time: 2019-11-05
# Author: Zachary 
# Name: MADDPG_torch
# File func: main func
import os
import time
import torch
import pickle
import argparse
import numpy as np
import torch.nn as nn
import torch.optim as optim

from arguments import parse_args

from utils.replay_buffer import ReplayBuffer
from utils.model import openai_actor, openai_critic

from envs.level0.explore_wrapper import explore_wrapper


def get_trainers(env, num_adversaries, obs_shape_n, action_shape_n, action_bound, arglist):
    """
    init the trainers or load the old model
    """
    actors_cur = [None for _ in range(env.n)]
    critics_cur = [None for _ in range(env.n)]
    actors_tar = [None for _ in range(env.n)]
    critics_tar = [None for _ in range(env.n)]
    optimizers_c = [None for _ in range(env.n)]
    optimizers_a = [None for _ in range(env.n)]
    

    # Note: if you need load old model, there should be a procedure for juding if the trainers[idx] is None
    for i in range(env.n):
        actors_cur[i] = openai_actor(obs_shape_n[i], action_shape_n[i], action_bound, arglist).to(arglist.device)
        critics_cur[i] = openai_critic(sum(obs_shape_n), sum(action_shape_n), arglist).to(arglist.device)
        actors_tar[i] = openai_actor(obs_shape_n[i], action_shape_n[i], action_bound, arglist).to(arglist.device)
        critics_tar[i] = openai_critic(sum(obs_shape_n), sum(action_shape_n), arglist).to(arglist.device)
        optimizers_a[i] = optim.Adam(actors_cur[i].parameters(), arglist.lr_a)
        optimizers_c[i] = optim.Adam(critics_cur[i].parameters(), arglist.lr_c)
    actors_tar = update_trainers(actors_cur, actors_tar, 1.0) # update the target par using the cur , tau = 1 : 제일 처음에는 복사해야 하므로
    critics_tar = update_trainers(critics_cur, critics_tar, 1.0) # update the target par using the cur , tau = 1 : 제일 처음에는 복사해야 하므로
    return actors_cur, critics_cur, actors_tar, critics_tar, optimizers_a, optimizers_c

def update_trainers(agents_cur, agents_tar, tau):
    """
    update the trainers_tar par using the trainers_cur
    This way is not the same as copy_, but the result is the same
    out:
    |agents_tar: the agents with new par updated towards agents_current
    """
    for agent_c, agent_t in zip(agents_cur, agents_tar):
        key_list = list(agent_c.state_dict().keys())
        state_dict_t = agent_t.state_dict()
        state_dict_c = agent_c.state_dict()
        for key in key_list:
            state_dict_t[key] = state_dict_c[key]*tau + \
                    (1-tau)*state_dict_t[key] 
        agent_t.load_state_dict(state_dict_t)
    return agents_tar

def agents_train(arglist, total_step, update_cnt, memory, obs_size, action_size, \
                actors_cur, actors_tar, critics_cur, critics_tar, optimizers_a, optimizers_c):
    """ 
    use this func to make the "main" func clean
    par:
    |input: the data for training
    |output: the data for next update
    """
    # update all trainers, if not in display or benchmark mode
    if total_step > arglist.learning_start_step and \
        (total_step - arglist.learning_start_step) % arglist.learning_fre == 0:
        if update_cnt == 0: print('\r=start training ...'+' '*100)
        # update the target par using the cur
        update_cnt += 1

        # update every agent in different memory batch
        for agent_idx, (actor_c, actor_t, critic_c, critic_t, opt_a, opt_c) in \
            enumerate(zip(actors_cur, actors_tar, critics_cur, critics_tar, optimizers_a, optimizers_c)):
            if opt_c == None: continue # jump to the next model update

            # sample the experience
            _obs_n_o, _action_n, _rew_n, _obs_n_n, _done_n = memory.sample( \
                arglist.batch_size, agent_idx) # Note_The func is not the same as others
                
            # --use the date to update the CRITIC
            rew = torch.tensor(_rew_n, device=arglist.device, dtype=torch.float) # set the rew to gpu
            done_n = torch.tensor(~_done_n, dtype=torch.float, device=arglist.device) # set the rew to gpu
            action_cur_o = torch.from_numpy(_action_n).to(arglist.device, torch.float)
            obs_n_o = torch.from_numpy(_obs_n_o).to(arglist.device, torch.float)
            obs_n_n = torch.from_numpy(_obs_n_n).to(arglist.device, torch.float)
            action_tar = torch.cat([a_t(obs_n_n[:, obs_size[idx][0]:obs_size[idx][1]]).detach() \
                for idx, a_t in enumerate(actors_tar)], dim=1)
            q = critic_c(obs_n_o, action_cur_o).reshape(-1) # q 
            q_ = critic_t(obs_n_n, action_tar).reshape(-1) # q_ 
            tar_value = q_*arglist.gamma*done_n + rew # q_*gamma*done + reward
            loss_c = torch.nn.MSELoss()(q, tar_value) # bellman equation
            opt_c.zero_grad()
            loss_c.backward()
            nn.utils.clip_grad_norm_(critic_c.parameters(), arglist.max_grad_norm)
            opt_c.step()


            # --use the data to update the ACTOR
            model_out, policy_c_new = actor_c( \
                obs_n_o[:, obs_size[agent_idx][0]:obs_size[agent_idx][1]], model_original_out=True) # There is no need to cal other agent's action
            # update the aciton of this agent
            action_cur_o[:, action_size[agent_idx][0]:action_size[agent_idx][1]] = policy_c_new 
            loss_pse = torch.mean(torch.pow(model_out, 2))
            loss_a = torch.mul(-1, torch.mean(critic_c(obs_n_o, action_cur_o)))
            loss_a_tot = 1e-2*loss_pse+loss_a

            opt_a.zero_grad()
            (loss_a_tot).backward()
            nn.utils.clip_grad_norm_(actor_c.parameters(), arglist.max_grad_norm)
            opt_a.step()

            # add tensorboard
            arglist.writer.add_scalar('loss_agent{}/Actor_regularization'.format(agent_idx), loss_pse.item(), update_cnt)
            arglist.writer.add_scalar('loss_agent{}/Actor'.format(agent_idx), loss_a.item(), update_cnt)
            arglist.writer.add_scalar('loss_agent{}/Critic'.format(agent_idx), loss_c.item(), update_cnt)


        # save the model to the path_dir ---cnt by update number
        if update_cnt > arglist.start_save_model and update_cnt % arglist.fre4save_model == 0:
            time_now = time.strftime('%y%m_%d%H%M')
            print('=time:{} step:{} saved!!'.format(time_now, total_step))
            model_file_dir = os.path.join(arglist.save_dir, '{}_{}'.format(arglist.scenario_name, total_step))
            if not os.path.exists(model_file_dir): # make the path
                os.mkdir(model_file_dir)
            for agent_idx, (a_c, a_t, c_c, c_t) in \
                enumerate(zip(actors_cur, actors_tar, critics_cur, critics_tar)):
                torch.save(a_c, os.path.join(model_file_dir, 'a_c_{}.pt'.format(agent_idx)))
                torch.save(a_t, os.path.join(model_file_dir, 'a_t_{}.pt'.format(agent_idx)))
                torch.save(c_c, os.path.join(model_file_dir, 'c_c_{}.pt'.format(agent_idx)))
                torch.save(c_t, os.path.join(model_file_dir, 'c_t_{}.pt'.format(agent_idx)))

        # update the tar par
        actors_tar = update_trainers(actors_cur, actors_tar, arglist.tau) 
        critics_tar = update_trainers(critics_cur, critics_tar, arglist.tau) 

    return update_cnt, actors_cur, actors_tar, critics_cur, critics_tar

def train(arglist):
    """
    init the env, agent and train the agents
    """
    """step1: create the environment """

    env_config = {
        "num_agents": 4,
        "obs_box_size": 50,
        "init_pos": ((140., 140.), (160., 140.), (140., 160.), (160., 160.)),
        # "init_pos": ((88, 69), (190, 120), (67, 220), (195, 220)),
        "dynamic_delta_t": 1.1
    }
    # 참고 : 'enemy_init_pos': ((88, 69), (190, 120), (67, 220), (195, 220))

    env = explore_wrapper(env_config)

    # Action space (2-dim, [가속도, 각가속도] )
    action_space = env.agents[0].action_space
    action_bound = [np.array([action_space[0].low[0], action_space[1].low[0]]), np.array([action_space[0].high[0], action_space[1].high[0]])] 

    
    print('=============================')
    print('=1 Env {} is right ...'.format(arglist.scenario_name))
    print('=============================')

    """step2: create agents"""

    obs_shape_n = [12 for _ in range(env.n)] # [global x,y] + [cosyaw,sinyaw] + [rel x,y]*4
    action_shape_n = [2 for _ in range(env.n)] # [가속도, 각가속도]
    num_adversaries = None # there is no adversaries in aicrew

    actors_cur, critics_cur, actors_tar, critics_tar, optimizers_a, optimizers_c = \
        get_trainers(env, num_adversaries, obs_shape_n, action_shape_n, action_bound, arglist)

    memory = ReplayBuffer(arglist.memory_size)
    
    print('=2 The {} agents are inited ...'.format(env.n))
    print('=============================')

    """step3: init the pars """
    obs_size = []
    action_size = []
    episode_idx = 0
    total_step = 0
    update_cnt = 0
    

    # Calculate observation size for MARL
    head_o, head_a, end_o, end_a = 0, 0, 0, 0
    for obs_shape, action_shape in zip(obs_shape_n, action_shape_n):
        end_o = end_o + obs_shape
        end_a = end_a + action_shape 
        range_o = (head_o, end_o)
        range_a = (head_a, end_a)
        obs_size.append(range_o)
        action_size.append(range_a)
        head_o = end_o
        head_a = end_a

    # obs_size [(0, 12), (12, 24), (24, 36), (36, 48)]
    # action_size [(0, 2), (2, 4), (4, 6), (6, 8)]

    print('=3 starting iterations ...')
    print('=============================')

    reset_arg = {'episode': 0}

    # Training
    for _ in range(arglist.max_episode):
        
        episode_idx += 1

        episode_reward = 0
        agent_rewards = [0 for _ in range(env.n)] 

        obs_n = env.reset(**reset_arg)
        reset_arg['episode'] = episode_idx
        
        for episode_steps in range(arglist.per_episode_max_len):
            # get action
            action_n = [agent(torch.from_numpy(obs).to(arglist.device, torch.float)).detach().cpu().numpy() \
                for agent, obs in zip(actors_cur, obs_n)]

            # interact with env
            new_obs_n, rew_n, done_n, info_n = env.step(action_n)
            # env.render()

            # save the experience
            memory.add(obs_n, np.concatenate(action_n), rew_n , new_obs_n, done_n)
            episode_reward += np.sum(rew_n)
            for i, rew in enumerate(rew_n): agent_rewards[i] += rew

            # train our agents 
            update_cnt, actors_cur, actors_tar, critics_cur, critics_tar = agents_train(\
                arglist, total_step, update_cnt, memory, obs_size, action_size, \
                actors_cur, actors_tar, critics_cur, critics_tar, optimizers_a, optimizers_c)

            # update the obs_n
            total_step += 1
            obs_n = new_obs_n
            done = done_n
            terminal = (episode_steps >= arglist.per_episode_max_len-1)

            if done or terminal:
                break


        # Evaluation
        if episode_idx % 20 == 0 :
            # arglist.video.init(enabled=True)
            episode_reward = 0
            agent_rewards = [0 for _ in range(env.n)] 

            for _ in range(arglist.num_epi_eval):
                obs_n = env.reset(**reset_arg)

                for episode_steps in range(arglist.per_episode_max_len):
                    # get action
                    action_n = [agent(torch.from_numpy(obs).to(arglist.device, torch.float)).detach().cpu().numpy() \
                        for agent, obs in zip(actors_cur, obs_n)]

                    # interact with env
                    new_obs_n, rew_n, done_n, info_n = env.step(action_n)

                    # rendering
                    # env.render()
                    # arglist.video.record(env.render(mode='rgb_array'))

                    # save the experience
                    episode_reward += np.sum(rew_n)
                    for i, rew in enumerate(rew_n): agent_rewards[i] += rew

                    # update the obs_n
                    obs_n = new_obs_n
                    done = done_n
                    terminal = (episode_steps >= arglist.per_episode_max_len-1)

                    if done or terminal:
                        break
            
            # add tensorboard
            arglist.writer.add_scalar('evaluation/reward', episode_reward/arglist.num_epi_eval, episode_idx)

            # print
            print("episode : {}, reward : {}".format(episode_idx, episode_reward/arglist.num_epi_eval))

            # arglist.video.save('test_{}.mp4'.format(episode_idx))
            # arglist.video.init(enabled=False)

if __name__ == '__main__':

    arglist = parse_args()
    train(arglist)
