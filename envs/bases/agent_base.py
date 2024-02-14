import gym
import numpy as np
from enum import IntEnum
from envs.bases.geo_base import geo_data_type
from envs.bases.DynK1A1acc import DynK1A1acc
from envs.bases.Dynamics import Dynamics #20231228
import math

class FightModeMoveActions(IntEnum):
    # NOTE: the first 4 actions also match the directional vector
    right = 0  # Move right
    down = 1  # Move down
    left = 2  # Move left
    up = 3  # Move up
    stop = 4  # stop
    no_op = 5  # dead


class FightModeFightActions(IntEnum):
    search = 0  # find enemy
    aiming = 1  # move turret
    fire = 2  # Fire
    no_op = 3  # dead


class AgentInterface:

    class FightModeMoveActions(IntEnum):
        # NOTE: the first 4 actions also match the directional vector
        right = 0  # Move right
        down = 1  # Move down
        left = 2  # Move left
        up = 3  # Move up
        fire = 4  # Fire
        aiming = 5  # move turret
        stop = 6  # stop
        search = 7  # find enemy
        no_op = 8  # dead

    def __init__(
            self,
            dyn_params_,
            geo_data_,
            agent_id=0,
            health=100,  # max 100 int
            has_weapon=True,  # bool
            num_ammunition=70,  # int
            view_range=2000,  # meter float
            weight=23000,  # Kg
            max_speed=65,  # 60 km/h
            max_force=100,  # Torque N.m
            max_turn_rate=0.5,  # 0.5 RPM
            rough_road_degrade_rate=80,  # 야지 주행능력 80%
            blue_red_team=0,  # 0 blue 1red
            obs_box_size=50,
            map_width=500,
            init_position=(0., 0.),
            dyna_delta_t=0.1,

    ):
        self.dynamic_parameter = dyn_params_
        self.geo_grid_data = geo_data_
        self.init_position = init_position
        self.init_num_ammunition = num_ammunition

        self.map_width = map_width
        self.obs_box_size = obs_box_size
        self.agent_id = agent_id
        self.health = health
        self.has_weapon = has_weapon
        self.num_ammunition = num_ammunition
        self.view_range = view_range
        self.weight = weight
        self.max_speed = max_speed
        self.max_force = max_force
        self.max_turn_rate = max_turn_rate
        self.rough_road_degrade_rate = rough_road_degrade_rate
        self.blue_red_team = blue_red_team
        self.position = init_position
        self.dynamic_delta_t = dyna_delta_t

        # self.map_width = map_width
        # self.map_height = map_height

        self.done = False
        self.behavior = 0
        self.closest_detected_enemy_dist = 0.0
        self.aiming_target_id = 0
        self.most_far_placed_enemy_dist = 0.0  # regardless of detected
        self.turret_abs_direction = 0.0  # 0.0 red means 3 o'clock, clock-wise
        self.current_speed = 0.0
        self.current_orientation = 0.0 # radian +X(3 o'clock) is 0.0, clock-wise
        self.list_detected_enemy = []

        self.reward = np.zeros(2)  # explore/fight mode 공통으로 사용하려면..
        self.active = True

        self.action_space = gym.spaces.Tuple(
            (
                gym.spaces.Box(self.dynamic_parameter['lim_a_min'], self.dynamic_parameter['lim_a_max'], dtype=float),  # 가속 m/s2 전장탐색/교전 공통
                gym.spaces.Box(self.dynamic_parameter['lim_ar_min'], self.dynamic_parameter['lim_ar_max'], dtype=float),  # 각가속 deg/s2  전장탐색용/교전 공
                gym.spaces.Discrete(len(FightModeMoveActions)),  # 교전 시 action  <-- deprecated 2023.11.09 Group chatting. 교전도 가속도 값을 보내오기로 함. 기존 코드 호환을 위해 설려 둠. 삭제할 경우, flag 참조하는 곳의 index를 4-->3 혹은 len(self.action)-1 로 수정해야 함.
                gym.spaces.Discrete(len(FightModeFightActions)),  # 교전 시 action통
                gym.spaces.Discrete(2),  # 탐색 0 or 교전 모드 1  flag
            )
        )

        # observation은 계속 추가해 갈 것임. 20230820 hoyabina
        # binary 여러개 말고 Box로.. 아래 Dict 코드는 사용되지 않고 그 아래 Box들이 사용됨. 20230825 hoyabina
        self.observation_space_not_use = gym.spaces.Dict({
                'disable': gym.spaces.MultiBinary([obs_box_size,obs_box_size], seed=42),  # R
                'obstacle': gym.spaces.MultiBinary([obs_box_size,obs_box_size], seed=42),  # G
                'runway': gym.spaces.MultiBinary([obs_box_size,obs_box_size], seed=42),  # B
                'object': gym.spaces.MultiBinary([obs_box_size,obs_box_size], seed=42),  # O
                'height': gym.spaces.Box(0.0, 256.0, shape=(obs_box_size,obs_box_size), dtype=np.float32)  # Height
            }
        )

        # 이 형태로 observation_space를 임시로 적용함. 20230825 hoyabina
        # 위의 Dict에서 R, G, B, Agent, Enemy를 각각 MultiBinary로 만들던 것을 Box 하나로 합침. geo_grid와 type을 일치시킴.
        self.observation_space_rgbae = gym.spaces.Box(0, 1, shape=(obs_box_size,obs_box_size,5), dtype=np.uint16)  # rgbae
        # objects. rendering용.  object의 크기만큼 격자를 RGB로 채운다. agent(255,255,255) enemy(255,0,255). object들끼지 겹치지 않는 특성.
        self.observation_space_object = gym.spaces.Box(0, 255, shape=(obs_box_size,obs_box_size,3), dtype=np.uint8)  # object state
        # Height
        self.observation_space_height = gym.spaces.Box(0.0, 256.0, shape=(obs_box_size,obs_box_size), dtype=np.float32)  # Height
        # # objects position
        # self.observation_space_object_pos = gym.spaces.Box(0.0, map_width, shape=(8,2), dtype=np.float32)  # objects position
        # explore memory data 처음 탐색한 곳인지 기억용
        self.observation_space_explore = gym.spaces.Box(0, 1, shape=(map_width, map_width), dtype=np.uint8)  # global
        # 교전용. Ray-cast 결과. 2023.11.11 hoyabina
        self.observation_space_visible = gym.spaces.Box(0, 2, shape=(view_range[0]*2, view_range[0]*2), dtype=np.uint8)  # global
        # RGBAE 교전용. shape을 view_range에 맞춤.
        self.observation_space_rgbae_fight = gym.spaces.Box(0, 1, shape=(view_range[0]*2, view_range[0]*2,5), dtype=np.uint16)  # Height
        # objects.  교전용. shape을 view_range에 맞춤.
        self.observation_space_object_fight = gym.spaces.Box(0, 255, shape=(view_range[0]*2, view_range[0]*2,3), dtype=np.uint8)  # Height
        # Height 교전용. shape을 view_range에 맞춤.
        self.observation_space_height_fight = gym.spaces.Box(0.0, 256.0, shape=(view_range[0]*2, view_range[0]*2), dtype=np.float32)  # Height
        # Dict 타입을 써야 random access를 할 수 있다.
        self.observation_space = gym.spaces.Dict(
            {
                'rgbae': self.observation_space_rgbae,
                'objects': self.observation_space_object,
                'height': self.observation_space_height,
                'explore': self.observation_space_explore,
                'visible': self.observation_space_visible,
                'rgbae_fight': self.observation_space_rgbae_fight,
                'objects_fight': self.observation_space_object_fight,
                'height_fight': self.observation_space_height_fight,
            }
        )

        self.action = self.action_space.sample()
        self.observation = self.observation_space.sample()
        self.explore_memory = np.zeros((map_width, map_width), dtype=np.uint8)
        # self.class_dynamic = DynK1A1acc(self.dynamic_delta_t, 1, 1, self.init_position[0], self.init_position[1], self.current_orientation)
        self.class_dynamic_new = Dynamics(
            self.geo_grid_data,
            self.dynamic_delta_t,
            self.init_position[0],
            self.init_position[1],
            0,
            0,
            0.,
            0,
            0,
            self.dynamic_parameter
        )

        print(init_position)

    # def set_obs_data(self, obs_):
    #     print(obs_)

    def reset(self):
        self.done = False
        self.active = True
        self.num_ammunition = self.init_num_ammunition
        self.behavior = 0
        self.closest_detected_enemy_dist = 0.0
        self.closest_detected_enemy_id = ''
        self.aiming_target_id = 0
        self.most_far_placed_enemy_dist = 0.0
        self.turret_abs_direction = 0.0  # 0.0 deg means 12 o'clock, counter-clock-wise
        self.position = self.init_position
        self.current_speed = 0.0
        self.current_orientation = 0.0
        self.list_detected_enemy = []
        self.reward[0] = -1
        self.reward[1] = 0
        self.observation['rgbae'].fill(0)
        self.observation['objects'].fill(0)
        self.observation['height'].fill(0)
        self.observation['explore'].fill(0)
        self.observation['visible'].fill(0)
        self.observation['rgbae_fight'].fill(0)
        self.observation['objects_fight'].fill(0)
        self.observation['height_fight'].fill(0)
        # self.class_dynamic = DynK1A1acc(self.dynamic_delta_t, 1, 1, self.init_position[0], self.init_position[1], self.current_orientation)
        self.class_dynamic_new = Dynamics(
            self.geo_grid_data,
            self.dynamic_delta_t,
            self.init_position[0],
            self.init_position[1],
            0,
            0,
            0,
            0,
            0,
            self.dynamic_parameter
        )
        
        # 시작 지점을 이미 탐색한 것으로 설정.
        self.observation['explore'][int(self.position[0])][int(self.position[1])] = 1

    def do_dynamic(self, ax_, alpha_):
        # return self.class_dynamic.sim_once(ax_, alpha_)
        return self.class_dynamic_new.step(ax_, alpha_)

    def do_move(self):
        #x, y, yaw, vx, yaw_rate, ax, alpha = self.do_dynamic(1.5, 12.1)
        x, y, yaw, vx, yaw_rate, ax, alpha = self.do_dynamic(self.action[0][0], self.action[1][0])
        # print(self.agent_id, "action[0]", self.action[0], "action[1]", self.action[1], x, y, yaw, yaw_rate, ax, alpha)
        # if self.agent_id == ['agent_1']:
        #     print("action value  ax alpha: ", self.action[0][0], self.action[1][0])

        if x < 0:
            x = 0
        if x >= self.map_width:
            x = self.map_width-1
        if y < 0:
            y = 0
        if y >= self.map_width:
            y = self.map_width-1

            # self.active = False # map을 벗어나면 active 종료..  20231109 active 조건 삭제. hoyabina
            #  todo reward <-- No. activation is not factor for reward

        # self.position = (x, y)  # 20240102 hoyabina, this wss a bug..
        self.position_desired = (x, y)
        # print("dyn input result: ", self.agent_id, self.action[0][0], self.action[1][0], x, y)
        self.current_orientation = yaw

        # todo check position is where the obstacle is. <-- env_base step 에서 check함. refactory 요.. env_base-->agent_base <-- 20231118 done

    def do_step(self, i, agents, enemies, bresenham_class):
        # 매 step 시작할 때 agent의 reward 중 explore 관련 reward마 초기화.
        self.reward[0] = -1
        # self.reward[1] = 0
        # self.list_detected_enemy = []  # 이것은 매 step 초기화하면 안됨. done이 성립할 수 없게 됨.

        if len(self.action) == 5:  # validate num of arguments

            # 0. move first. same way in explore, fight
            _prev_pos = self.position
            self.do_move()  # move first

            # agent별 step 결과로 이동된 위치가 장애물인지를 판단한다. 장매물이면 set active FALSE <-- deprecated
            # 20231101 hoyabina agent의 position은 timestep의 크기에 따라서 이전 position에서 보폭이 클 수 있으며
            # 이것 때문에 장애물을 훌쩍 뛰어 넘을 수 있다.
            # 이전 position 현재 position 사이에 장애물이 있는지 판단하여야 하며 장애물이 있을 시,
            # 1. 장애물과 만나는 위치에 고정
            _can_see, (obstacle_x, obstacle_y) = bresenham_class.can_see_eachother( \
            # _prev_pos[0], _prev_pos[1], self.position_desired[0], self.position_desired[1])
            int(_prev_pos[0]), int(_prev_pos[1]), int(self.position_desired[0]), int(self.position_desired[1]))

            # print("can_see_eachother result: ", _can_see, self.position_desired[0], self.position_desired[1], obstacle_x, obstacle_y)
            # 20231227 _can_see == False 이면 장애물을 만난 것임. 이때 dynamic update에 collision_flag 값 = true 룰 지정해 줌.

            if _can_see is True:
                self.class_dynamic_new.update(False, self.position_desired[0], self.position_desired[1], 0.0)
                self.position = (self.position_desired[0], self.position_desired[1])  # 20231101 이후에는 동력학 결과로 얻어진 position으로 이동중, 장애물과 만나는 지점을 position으로 함
            else:
                self.class_dynamic_new.update(True, obstacle_x, obstacle_y, 0.0)
                self.position = (obstacle_x, obstacle_y)  # 20231101 이후에는 동력학 결과로 얻어진 position으로 이동중, 장애물과 만나는 지점을 position으로 함

            # print("obstacle result: ", self.agent_id, _can_see, obstacle_x, obstacle_y)

            # 장애물에 부딛혔는지 Debug
            # if obstacle_x != _prev_pos[0] or obstacle_y != _prev_pos[1]:
            #     print("while moving, agent meets the obstacle: dyn result", self.position[0], self.position[1])
            #     print("while moving, agent meets the obstacle: dyn result", self.position[0], self.position[1])

            # 2. Reward after moving
            # Reward in explore
            if self.action[4] == 0:  # explore mode, 0:explore 1:fight
                # move 완료한 position이 처음 와 본 위치이면 보상을 주고 와 보았다고 기억.
                # todo ??? 기속환경에서 이전 위치로부터 move 완료한 위치까지 jump를 할텐데.. 그 중간 위치들에 대한 보상과 기억은 ???
                if self.observation['explore'][int(self.position[0])][int(self.position[1])] == 0:  # when move to un-explored area
                    self.set_reward_for_explore(0)  # then set explore-reward 0

                self.observation['explore'][int(self.position[0])][int(self.position[1])] = 1  # memorize explored area

                # agent, enemy의 move를 마쳤으면 LOS check.
                self.closest_detected_enemy_dist = 10000
                self.most_far_placed_enemy_dist = 0
                for enemy in enemies:
                    if enemy.active is True:
                        agent_pos = self.position
                        enemy_pos = enemy.position
                        distance = math.dist(list(agent_pos), list(enemy_pos))
                        # visibility와 무관하게 가장 멀리 있는 적 객체까지의 거리를 파악한다.
                        self.most_far_placed_enemy_dist = distance if distance > self.most_far_placed_enemy_dist\
                            else self.most_far_placed_enemy_dist

                        _can_see, (obstacle_x, obstacle_y) = bresenham_class.can_see_eachother(\
                                int(agent_pos[0]), int(agent_pos[1]), int(enemy_pos[0]), int(enemy_pos[1]));

                        if _can_see:
                            # detected 된 적 객체들 중에 가장 가까운 거리를 파악한다.
                            if distance < self.closest_detected_enemy_dist:
                                self.closest_detected_enemy_dist = distance
                                self.closest_detected_enemy_id = enemy.enemy_id

                            # Detected enemy 정보를 모든 agent와 공유한다.
                            for k, agent_for_reward in enumerate(agents):  # actions)):
                                # agent_for_reward = self.agents[k]
                                if i == k:  # 직접 발견한 agent는 보상 2
                                    agent_for_reward.append_detected_enemy(enemy, 2)
                                else:  # 나머지 agent는 보상 1
                                    agent_for_reward.append_detected_enemy(enemy, 1)

# Debug
                                # print("Reward.. id:", agent_for_reward.agent_id, " active:", agent_for_reward.active,\
                                # " reward:", agent_for_reward.reward)

                            # print("------------------Can See num", agent.num_detected_enemy(), agent.agent_id,\
                            #      agent.list_detected_enemy, distance)

                        else:
                            # BUG..!! 적이 안보이는 순간, detected enemy에서 삭제하면 다시 그 적이 보이는 순간 보상을 받게 된다. 삭제
                            # agent.remove_detected_enemy(enemy)
                            continue

            else: # fight mode 0:explore 1:fight
                agent_pos = self.position
                self.observation['visible'].fill(0)

                # 0. Do ray-casting and update observation['visible']
                # last parameter : degree step-size.
                # bresenham_class.do_raytracing(int(agent_pos[0]), int(agent_pos[1]), self.view_range[0], \
                #                               self.observation['visible'], 2)

                pass
                # print(self.agent_id, "Fight mode action", self.action[4])

        else:
            print(self.agent_id, "wrong action space count..")

    def get_observation_space(self):
        return self.observation_space

    def add_reward_for_explore(self, value):
        self.reward[0] += value

    def set_reward_for_explore(self, value):
        self.reward[0] = value

    def add_reward_for_detect_enemy(self, value):
        self.reward[1] += value

    def do_reward(self):
        return

    @property
    def health(self):
        return self._health

    @health.setter
    def health(self, health_):
        self._health = health_

    @property
    def behavior(self):
        return self._behavior

    @behavior.setter
    def behavior(self, behavior_):
        self._behavior = behavior_

    @property
    def closest_detected_enemy_dist(self):
        return self._closest_detected_enemy_dist

    @closest_detected_enemy_dist.setter
    def closest_detected_enemy_dist(self, dist_):
        self._closest_detected_enemy_dist = dist_

    @property
    def closest_detected_enemy_id(self):
        return self._closest_detected_enemy_id

    @closest_detected_enemy_id.setter
    def closest_detected_enemy_id(self, id_):
        self._closest_detected_enemy_id = id_

    @property
    def aiming_target_id(self):
        return self._aiming_target_id

    @aiming_target_id.setter
    def aiming_target_id(self, id_):
        self._aiming_target_id = id_

    @property
    def most_far_placed_enemy_dist(self):
        return self._most_far_placed_enemy_dist

    @most_far_placed_enemy_dist.setter
    def most_far_placed_enemy_dist(self, dist_):
        self._most_far_placed_enemy_dist = dist_

    @property
    def turret_abs_direction(self):
        return self._turret_abs_direction

    @turret_abs_direction.setter
    def turret_abs_direction(self, deg_):
        self._turret_abs_direction = deg_

    @property
    def position(self):
        return self._position

    @position.setter
    def position(self, pos_):
        self._position = pos_

    @property
    def current_speed(self):
        return self._current_speed

    @current_speed.setter
    def current_speed(self, speed_):
        self._current_speed = speed_

    @property
    def current_orientation(self):
        return self._current_orientation

    @current_orientation.setter
    def current_orientation(self, orientation_):
        self._current_orientation = orientation_

    @property
    def active(self):
        return self._active

    @active.setter
    def active(self, act_):
        self._active = act_

    def num_detected_enemy(self):
        return len(self.list_detected_enemy)

    def append_detected_enemy(self, enemy, reward_val):
        if enemy.enemy_id not in self.list_detected_enemy:
            self.list_detected_enemy.append(enemy.enemy_id)
            self.add_reward_for_detect_enemy(reward_val) # 기존에 찾아낸 적군이 아닐 경우, 보상 추가.. todo ??? 내가 아닌 다른 아군이 찾아낸 적군을 내가 다시 찾아 냈을 경우에는 보상이 없는가?

    # 시야에서 사리지면 detected 되지 않은 것으로 변경한다.
    # 이후, 다시 detected 되면 또 보상을 하게 되는데 맞는가?
    def remove_detected_enemy(self, enemy):
        if enemy.enemy_id in self.list_detected_enemy:
            self.list_detected_enemy.remove(enemy.enemy_id)




