# Time: 2024/02/15
# Author: JH
# Name: AIcrew-MADDPG

import datetime
import time
import torch
import argparse
from torch.utils.tensorboard import SummaryWriter
from utils.video import VideoRecorder

# Device
device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
time_now = time.strftime('%y%m_%d%H%M')

# Tesnorboard
writer = SummaryWriter('runs/{}_MADDPG_{}'.format(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"), "AI-crew"))
log_dir = 'runs/{}_MADDPG_{}'.format(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"), "AI-crew")

# For video
video_directory = './video/{}'.format(datetime.datetime.now().strftime("%H:%M:%S %p"))
video = VideoRecorder(dir_name = video_directory)


def parse_args():
    parser = argparse.ArgumentParser("reinforcement learning experiments for multiagent environments")

    # environment
    parser.add_argument("--scenario_name", type=str, default="aicrew", help="name of the scenario script")
    parser.add_argument("--start_time", type=str, default=time_now, help="the time when start the game")
    parser.add_argument("--per_episode_max_len", type=int, default=200, help="maximum episode length")
    parser.add_argument("--max_episode", type=int, default=10000, help="maximum episode length") # 200 * 10000 = 2,000,000, 2M이면 충분
    parser.add_argument("--num-adversaries", type=int, default=1, help="number of adversaries")

    # core training parameters
    parser.add_argument("--writer", default=writer, help="tensorboard writer")
    parser.add_argument("--video", default=video, help="video")
    parser.add_argument("--device", default=device, help="torch device")
    parser.add_argument("--learning_start_step", type=int, default=50000, help="learning start steps")
    parser.add_argument("--max_grad_norm", type=float, default=0.5, help="max gradient norm for clip")
    parser.add_argument("--learning_fre", type=int, default=100, help="learning frequency")
    parser.add_argument("--tau", type=int, default=0.01, help="how depth we exchange the params of the nn")
    parser.add_argument("--lr_a", type=float, default=1e-2, help="learning rate for adam optimizer")
    parser.add_argument("--lr_c", type=float, default=1e-2, help="learning rate for adam optimizer")
    parser.add_argument("--gamma", type=float, default=0.97, help="discount factor")
    parser.add_argument("--batch_size", type=int, default=1256, help="number of episodes to optimize at the same time")
    parser.add_argument("--memory_size", type=int, default=1e6, help="number of data stored in the memory")
    parser.add_argument("--num_units_openai", type=int, default=128, help="number of units in the mlp")

    # checkpointing
    parser.add_argument("--fre4save_model", type=int, default=1000, help="the number of the episode for saving the model")
    parser.add_argument("--start_save_model", type=int, default=100, help="the number of the episode for saving the model")
    parser.add_argument("--old_model_name", type=str, default="saved_models/", \
            help="directory in which training state and model are loaded")
    parser.add_argument("--log_dir", type=str, default=log_dir, help="directory in which model_dir should be saved")

    # evaluation
    parser.add_argument("--num_epi_eval", type=int, default=1, help="number of episode")

    
    return parser.parse_args()
