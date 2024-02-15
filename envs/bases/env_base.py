import gym
from gym import utils
import numpy as np
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import pygame
from envs.bases.bresenham import Bresenham
import math
import os

from pygame import gfxdraw

steps = 0

def get_config(env_config, randomize_seed=True):
    # possible_envs = {k:v for k,v in globals().items() if inspect.isclass(v) and issubclass(v, MultiGridEnv)}

    # env_class = possible_envs[env_config['env_class']]

    env_kwargs = {k: v for k, v in env_config.items() if k != 'env_class'}
    # if randomize_seed:
    #    env_kwargs['seed'] = env_kwargs.get('seed', 0) + random.randint(0, 1337*1337)

    # return env_class(**env_kwargs)
    # print(env_kwargs)
    return env_kwargs


class tmps_env_base(gym.Env):

    def __init__(
            self,
            configs,
            width=None,
            height=None,
    ):


        self.screen_width_margin = 1000
        self.screen_height_margin = 400

        if configs['map_width'] is not None:
            assert width == None and height == None
            width, height = configs['map_width'], configs['map_height']

        self.width = width
        self.height = height
        self.geo_grid_data = configs['geo_grid_data']
        self.osb_box_size = configs['obs_box_size']

        self.agents = []
        for agent in configs['agents']:
            self.add_agent(agent)

        self.enemies = []
        for enemy in configs['enemies']:
            self.enemies.append(enemy)

        self.action = 0

        # self.reset()
        self.bresenham_class = Bresenham(self.geo_grid_data)

        # PyGame setting
        pygame.init()
        pygame.display.init()

        self.render_data = np.zeros((self.width + self.screen_width_margin, 300 + self.screen_height_margin, 3)) #render size
        print("render_data shape", self.render_data.shape)
        self.render_data[0:300, 0:300] = np.array(self.geo_grid_data.get_render_image())
        print("render_data shape2", self.render_data.shape)

        pygame.font.init()
        self.my_font = pygame.font.SysFont('Comic Sans MS', 30)

        self.test_num = 0

        self.screen = pygame.display.set_mode((self.width + self.screen_width_margin, self.height + self.screen_height_margin))  # , pygame.RESIZABLE)  # 창만 커진다. #render size

    def seed(self, seed=1337):
        # Seed the random number generator
        self.np_random, _ = gym.utils.seeding.np_random(seed)
        return [seed]

    @property
    def action_space(self):                    # 장애물 시작점까지 보이도록

        return gym.spaces.Tuple(
            [agent.action_space for agent in self.agents]
        )

    @property
    def observation_space(self):
        return gym.spaces.Tuple(
            [agent.observation_space for agent in self.agents]
        )

    @property
    def num_agents(self):
        return len(self.agents)

    def add_agent(self, agent_interface):
        print("add_agent")
        self.agents.append(agent_interface)
        # if isinstance(agent_interface, dict):
        #     self.agents.append(GridAgentInterface(**agent_interface))
        # elif isinstance(agent_interface, GridAgentInterface):
        #     self.agents.append(agent_interface)
        # else:
        #     raise ValueError(
        #         "To add an agent to a marlgrid environment, call add_agent with either a GridAgentInterface object "
        #         " or a dictionary that can be used to initialize one.")

    def reset(self, **kwargs):
        global steps
        steps = 0
        # print('env_base reset')
        for agent in self.agents:
            agent.reset()
        for enemy in self.enemies:
            enemy.reset()

        self.render_data = np.zeros((self.width + self.screen_width_margin, 300 + self.screen_height_margin, 3)) #render size
        self.geo_grid_data.reset()

        self.episode_num = kwargs['episode']
        self.text_out = "Episode " + str(self.episode_num)
        self.text_surface = self.my_font.render(self.text_out, False, (255, 255, 255))

        obs = self.gen_obs()
        # print(obs)
        return obs

    def gen_obs_grid(self, agent):
        print("gen_obs_grid")
        # If the agent is inactive, return an empty grid and a visibility mask that hides everything.
        # if not agent.active:
        #     # # below, not sure orientation is correct but as of 6/27/2020 that doesn't matter because
        #     # # agent views are usually square and this grid won't be used for anything.
        #     grid = MultiGrid((agent.view_size, agent.view_size), orientation=agent.dir + 1)
        #     vis_mask = np.zeros((agent.view_size, agent.view_size), dtype=np.bool)
        #     return grid, vis_mask
        #
        # topX, topY, botX, botY = agent.get_view_exts()
        #
        # grid = self.grid.slice(
        #     topX, topY, agent.view_size, agent.view_size, rot_k=agent.dir + 1
        # )

        # Process occluders and visibility
        # Note that this incurs some slight performance cost
        # vis_mask = agent.process_vis(grid.opacity)
        #
        # # Warning about the rest of the function:
        # #  Allows masking away objects that the agent isn't supposed to see.
        # #  But breaks consistency between the states of the grid objects in the parial views
        # #   and the grid objects overall.
        # if len(getattr(agent, 'hide_item_types', [])) > 0:
        #     for i in range(grid.width):
        #         for j in range(grid.height):
        #             item = grid.get(i, j)
        #             if (item is not None) and (item is not agent) and (item.type in agent.hide_item_types):
        #                 if len(item.agents) > 0:
        #                     grid.set(i, j, item.agents[0])
        #                 else:
        #                     grid.set(i, j, None)
        #
        # return grid, vis_mask

    def gen_agent_obs(self, agent):
        # way 1
        self.geo_grid_data.copy_abs_box(agent)

    def gen_obs(self):
        return [self.gen_agent_obs(agent) for agent in self.agents]

    def __str__(self):
        return "tmps_env_base"
        # return self.grid.__str__()

    # def check_agent_position_integrity(self, title=''):
    #     '''
    #     This function checks whether each agent is present in the grid in exactly one place.
    #     This is particularly helpful for validating the world state when ghost_mode=False and
    #     agents can stack, since the logic for moving them around gets a bit messy.
    #     Prints a message and drops into pdb if there's an inconsistency.
    #     '''
    #     agent_locs = [[] for _ in range(len(self.agents))]
    #     for i in range(self.grid.width):
    #         for j in range(self.grid.height):
    #             x = self.grid.get(i, j)
    #             for k, agent in enumerate(self.agents):
    #                 if x == agent:
    #                     agent_locs[k].append(('top', (i, j)))
    #                 if hasattr(x, 'agents') and agent in x.agents:
    #                     agent_locs[k].append(('stacked', (i, j)))
    #     if not all([len(x) == 1 for x in agent_locs]):
    #         print(f"{title} > Failed integrity test!")
    #         for a, al in zip(self.agents, agent_locs):
    #             print(" > ", a.color, '-', al)
    #         import pdb;
    #         pdb.set_trace()

    def step(self, actions):
        global steps
        steps += 1
        self.text_step_out = self.text_out + "  steps " + str(steps)
        self.text_surface = self.my_font.render(self.text_step_out, False, (255, 255, 255))

        # For objects position
        objects_position = np.zeros((len(self.agents) + len(self.enemies), 2), dtype=np.float32)

        # enemy의 상태를 global 에 갱신.. 현재는 움직이지 않지만 향후 움직임을 구현했을 때 필요함.
        for i, enemy in enumerate(self.enemies):
            # for i in range(len(self.enemies)):
            # for enemy in self.enemies:
            # enemy = self.enemies[i]
            self.put_enemy_to_global(enemy)
            objects_position[len(self.agents) + i] = (enemy.position[0], enemy.position[1])
            # objects_position[len(self.agents)*2 + i * 2 + 1] = enemy.position[1]

        # debug
        # print(objects_position)

        # agent별 step을 진행한다. agent의 진행은 dynamic에 의해 이동만 실행함.
        for i, agent in enumerate(self.agents):
            # for i in range(len(self.agents)):  # actions)):
            #     agent = self.agents[i]
            agent.action = actions[i]

            if agent.active is True:
                # agent의 현재 position으로 global_state에서 삭제
                self.remove_agent_from_global(agent)
                # 이동
                # _prev_pos = agent.position
                agent.do_step(i, self.agents, self.enemies, self.bresenham_class)

                # 20231111 move to agent_base
                # # agent별 step 결과로 이동된 위치가 장애물인지를 판단한다. 장매물이면 set active FALSE <-- deprecated
                # # 20231101 hoyabina agent의 position은 timestep의 크기에 따라서 이전 position에서 보폭이 클 수 있으며
                # # 이것 때문에 장애물을 훌쩍 뛰어 넘을 수 있다.
                # # 이전 position 현재 position 사이에 장애물이 있는지 판단하여야 하며 장애물이 있을 시,
                # # 장애물과 만나는 위치에 고정되어야 한다.
                #
                # _can_see, (obstacle_x, obstacle_y) = self.bresenham_class.can_see_eachother( \
                #     int(_prev_pos[0]), int(_prev_pos[1]), int(agent.position[0]), int(agent.position[1]))
                #
                # # 장애물에 부딛혔는지 Debug
                # if obstacle_x != _prev_pos[0] or obstacle_y != _prev_pos[1]:
                #     print("while moving, agent meets the obstacle: dyn result", agent.position[0], agent.position[1])
                #     print("while moving, agent meets the obstacle: dyn result", agent.position[0], agent.position[1])
                #
                # agent.position = (obstacle_x, obstacle_y) #20231101 이후에는 동력학 결과로 얻어진 position이 장애물과 만나는 지점을 position으로 함

                # _pos = agent.position #20231101 이전에는 동력학 결과로 얻어진 다음 position을 그대로 agent의 position으로 사용
                # _x = int(_pos[0])
                # _y = int(_pos[1])
                # _rgb_value = self.geo_grid_data.get_grid_RGB_property(_x, _y)
                # # print(_rgb_value)
                # # 20231109 아래 deactivate 조건 삭제.
                # # if _rgb_value[1] > 10 or _rgb_value[0] > 10:  # R:mountain, G:obstacle B:water G has value
                # #     agent.active = False # 이것은 장애물을 만나면 멈추는 것이다. todo False를 done 조건으로 할 것인가? <--20231109 No!!

                # agent의 이동한 position으로 global_state 갱신
                self.put_agent_to_global(agent)
                objects_position[i] = (agent.position[0], agent.position[1])
                # objects_position[i*2+1] = _y

        # # agent, enemy의 move를 마쳤으면 LOS check.  20231111 move to agent_base
        # # for agent in self.agents:
        # #     if agent.active is True:
        #         agent.closest_detected_enemy_dist = 10000
        #         agent.most_far_placed_enemy_dist = 0
        #         for enemy in self.enemies:
        #             if enemy.active is True:
        #                 agent_pos = agent.position
        #                 enemy_pos = enemy.position
        #                 distance = math.dist(list(agent_pos), list(enemy_pos))
        #                 # visibility와 무관하게 가장 멀리 있는 적 객체까지의 거리를 파악한다.
        #                 agent.most_far_placed_enemy_dist = distance if distance > agent.most_far_placed_enemy_dist\
        #                     else agent.most_far_placed_enemy_dist
        #
        #                 _can_see, (obstacle_x, obstacle_y) = self.bresenham_class.can_see_eachother(\
        #                         int(agent_pos[0]), int(agent_pos[1]), int(enemy_pos[0]), int(enemy_pos[1]));
        #
        #                 if _can_see:
        #                     # detected 된 적 객체들 중에 가장 가까운 거리를 파악한다.
        #                     if distance < agent.closest_detected_enemy_dist:
        #                         agent.closest_detected_enemy_dist = distance
        #                         agent.closest_detected_enemy_id = enemy.enemy_id
        #
        #                     # Detected enemy 정보를 모든 agent와 공유한다.
        #                     for k, agent_for_reward in enumerate(self.agents):  # actions)):
        #                         #agent_for_reward = self.agents[k]
        #                         if i == k:  # 직접 발견한 agent는 보상 2
        #                             agent_for_reward.append_detected_enemy(enemy, 2)
        #                         else:  # 나머지 agent는 보상 1
        #                             agent_for_reward.append_detected_enemy(enemy, 1)
        #
        #                         # print("Reward id:", agent_for_reward.agent_id, " active:", agent_for_reward.active,\
        #                         " reward:", agent_for_reward.reward)
        #
        #                     # print("------------------Can See num", agent.num_detected_enemy(), agent.agent_id,\
        #                     #      agent.list_detected_enemy, distance)
        #
        #                 else:
        #                     # BUG..!! 적이 안보이는 순간, detected enemy에서 삭제하면 다시 그 적이 보이는 순간 보상을 받게 된다. 삭제
        #                     # agent.remove_detected_enemy(enemy)
        #                     continue

        # Update global state
        observation_space_global = self.geo_grid_data.copy_global_box()

        # agent step을 완료한 후 observation 최신화
        for agent in self.agents:
            # if agent.active is True:   # agent의 active=False 상태를 obs render 창에 보여 주지 않더라.
            self.gen_agent_obs(agent)

        # Reward 연산,
        # explore - explore용과 적발견용 reward를 분리
        # 모든 step에 걸쳐 처음 가본 곳을 내딛는 순간 reward 0,
        # 적을 발견하면 모든 agent에게 +1, 발견을 한 agent에게는 추가로 +1
        step_rewards = [agent.reward for agent in self.agents]  # np.zeros((len(self.agents, )), dtype=np.float32)

        # test..
        done = False
        for agent in self.agents:
            num_detect = len(agent.list_detected_enemy)
            #print("num_detect", num_detect, " of ", len(self.enemies))
            if num_detect == len(self.enemies):  # 모든 적을 탐색하면 Done = True
                done = True
                break
            else:  # 적 탐색 정보를 모든 agent가 공유하므로 첫번째 agent만 보면 됨.
                break

        if done is not True:
            active_done = True
            for agent in self.agents:
                if agent.active is True:
                    active_done = False  # 하나라도 active이면 done=false
                    break

            # agent deactive 조건 삭제되었는지 확인
            if active_done is True:
                print("agent active condition is alive")

            done = active_done

        # observation array
        obs = [agent.observation for agent in self.agents]

        # 'global_state'는 전역 상태정보를 가지고 있음. 맵너비 x 맵높이 x 5 리스트, R G B A(Agent) E(Enemy)
        # 'object_pos'는 아군/적군의 위치 데이터를 담고 있음. (아군수 + 적군수) x 2 리스트,  x_pos, y_pos
        return obs, step_rewards, done, \
            {'agents': self.agents, 'global_state': observation_space_global, 'objects_pos': objects_position}

    def remove_agent_from_global(self, obj):
        self.geo_grid_data.remove_object_agent(obj)

    def put_agent_to_global(self, obj):
        self.geo_grid_data.put_object_agent(obj)

    def remove_enemy_from_global(self, obj):
        self.geo_grid_data.remove_object_enemy(obj)

    def put_enemy_to_global(self, obj):
        self.geo_grid_data.put_object_enemy(obj)

    def render(
            self,
            mode="human",
            close=False,
            highlight=True,
            # tile_size=TILE_PIXELS,
            show_agent_views=True,
            max_agents_per_col=3,
            agent_col_width_frac=0.3,
            agent_col_padding_px=2,
            pad_grey=100
    ):
        # print("render")
        """
        Render the whole-grid human view
        """

        # self.screen = pygame.display.set_mode((self.width + 300, self.height))  # , pygame.RESIZABLE)  # 창만 커진다.

        # self.render_data = np.array(self.geo_grid_data.get_render_image()) + np.zeros((300, 300, 3))
        self.render_data[0:300, 0:300] = np.array(self.geo_grid_data.get_render_image())

        for k in range(len(self.agents)):
            agent = self.agents[k]
            _pos = agent.position
            _x = int(_pos[0])
            _y = int(_pos[1])
            for i in range(-3, 3):
                for j in range(-3, 3):
                    x = max(0, min(_x + j, self.width - 1))
                    y = max(0, min(_y + i, self.width - 1))
                    if agent.active is True:
                        self.render_data[x, y] = [255, 255, 255]  # red dot in the center
                    else:
                        self.render_data[x, y] = [255, 64, 0]  # red dot in the center

            # observation window
            _obs = agent.observation['rgbae']  # geo enviroment
            _obs_objects = agent.observation['objects']  # dynamic objects
            self.render_data[self.width + k * (agent.obs_box_size + 20): \
                             self.width + k * (agent.obs_box_size + 20) + agent.obs_box_size, 0:agent.obs_box_size] \
                = _obs[:, :, 0:3] * 255 + _obs_objects[:, :, 0:3]



            _obs_visible = agent.observation['visible']  # view_range for fight
            _obs_fight = agent.observation['rgbae_fight']  # geo enviroment
            _obs_objects_fight = agent.observation['objects_fight']  # dynamic objects
            self.render_data[self.width + k * (agent.view_range[0]*2 + 20): \
                             self.width + k * (agent.view_range[0]*2 + 20) + agent.view_range[0]*2, agent.obs_box_size + 40: agent.obs_box_size + 40 + agent.view_range[0]*2] \
                = _obs_visible[:,:,None] * 255

            self.render_data[self.width + k * (agent.view_range[0]*2 + 20): \
                             self.width + k * (agent.view_range[0]*2 + 20) + agent.view_range[0]*2, \
            agent.obs_box_size + 40 + agent.view_range[0]*2 + 40 : agent.obs_box_size + 40 + agent.view_range[0]*2 + agent.view_range[0]*2 + 40] \
                = (_obs_fight[:, :, 0:3] * 255 + _obs_objects_fight[:, :, 0:3]) * _obs_visible[:, :, None]

        for enemy in self.enemies:
            # agent.agents = []
            _pos = enemy.position
            _x = int(_pos[0])
            _y = int(_pos[1])
            for i in range(-3, 3):
                for j in range(-3, 3):
                    x = max(0, min(_x + j, self.width - 1))
                    y = max(0, min(_y + i, self.width - 1))
                    self.render_data[x, y] = [255, 0, 255]  # red dot in the center

        self.test_num += 2

        if mode == "human":

            image = pygame.surfarray.make_surface(self.render_data)
            self.screen.blit(image, (0, 0))
            self.screen.blit(self.text_surface, (800, 30))
            for k in range(len(self.agents)):
                agent = self.agents[k]
                # agent 진행방향 표시
                pygame.draw.line(self.screen, (0, 255, 0), (agent.position[0], agent.position[1]), \
                                (agent.position[0]-math.sin(agent.current_orientation)*20, \
                                agent.position[1]+math.cos(agent.current_orientation)*20), 3)
                # pygame.draw.line(self.screen, (255, 255, 255), (self.width + k * (agent.obs_box_size + 20), 0),
                #                  (self.width + k * (agent.obs_box_size + 20) + agent.obs_box_size, 0),2)
                # pygame.draw.line(self.screen, (255, 255, 255), (self.width + k * (agent.obs_box_size + 20), agent.obs_box_size),
                #                  (self.width + k * (agent.obs_box_size + 20) + agent.obs_box_size, agent.obs_box_size),2)
                # pygame.draw.line(self.screen, (255, 255, 255), (self.width + k * (agent.obs_box_size + 20), 0),
                #                  (self.width + k * (agent.obs_box_size + 20) , agent.obs_box_size),2)
                # pygame.draw.line(self.screen, (255, 255, 255), (self.width + k * (agent.obs_box_size + 20)+ agent.obs_box_size, 0),
                #                  (self.width + k * (agent.obs_box_size + 20)+ agent.obs_box_size , agent.obs_box_size),2)
                # 전장탐색 obs창 테두리
                pygame.draw.rect(self.screen, (255, 255, 255), (self.width + k * (agent.obs_box_size + 20), 0,
                                agent.obs_box_size, agent.obs_box_size), 2)

                # pygame.draw.line(self.screen, (255, 255, 255), (self.width + k * (agent.view_range[0]*2 + 20), agent.obs_box_size + 40),
                #                  (self.width + k * (agent.view_range[0]*2 + 20) + agent.view_range[0]*2, agent.obs_box_size + 40),2)
                # pygame.draw.line(self.screen, (255, 255, 255), (self.width + k * (agent.view_range[0]*2 + 20), agent.obs_box_size + 40 + agent.view_range[0]*2),
                #                  (self.width + k * (agent.view_range[0]*2 + 20) + agent.view_range[0]*2, agent.obs_box_size + 40 + agent.view_range[0]*2),2)
                # pygame.draw.line(self.screen, (255, 255, 255), (self.width + k * (agent.view_range[0]*2 + 20), agent.obs_box_size + 40),
                #                  (self.width + k * (agent.view_range[0]*2 + 20) , agent.obs_box_size + 40 + agent.view_range[0]*2),2)
                # pygame.draw.line(self.screen, (255, 255, 255), (self.width + k * (agent.view_range[0]*2 + 20)+ agent.view_range[0]*2, agent.obs_box_size + 40),
                #                  (self.width + k * (agent.view_range[0]*2 + 20)+ agent.view_range[0]*2 , agent.obs_box_size + 40 + agent.view_range[0]*2),2)

                # 교전 obs창 테두리 - LOS
                pygame.draw.rect(self.screen, (255, 255, 255), (self.width + k * (agent.view_range[0]*2 + 20), agent.obs_box_size + 40,
                                agent.view_range[0]*2, agent.view_range[0]*2), 2)

                # 교전 obs창 테두리 - 지형지물 x LOS
                pygame.draw.rect(self.screen, (255, 255, 255), (self.width + k * (agent.view_range[0]*2 + 20), agent.obs_box_size + 40 + agent.view_range[0]*2 + 40,
                                agent.view_range[0]*2, agent.view_range[0]*2), 2)

            pygame.display.update()

        if mode == "rgb_array":
            image = pygame.surfarray.make_surface(self.render_data)
            rgb_array = pygame.surfarray.array3d(image)

            # 이미지를 시계방향으로 -90도 돌립니다.
            rotated_image = np.rot90(rgb_array, k=3)

            # 이미지를 좌우대칭 시킵니다.
            flipped_image = np.fliplr(rotated_image)
            return flipped_image # (600, 300, 3)