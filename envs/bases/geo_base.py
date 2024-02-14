import numpy as np
from PIL import Image
import os
import gym
# import envs.bases.agent_base

# observation data type
geo_data_type = np.dtype([('r', np.uint8), ('g', np.uint8), ('b', np.uint8), ('h', np.int16), ('o', np.uint8)])
# global state data type
geo_map_type = np.dtype([('x', np.uint16), ('y', np.uint16), ('d', geo_data_type)])


class GeoGridData:

    def __init__(self, path_, terrain_rgb_, heightmap_):

        if terrain_rgb_ is None or heightmap_ is None:
            return

        cwd = os.getcwd()

        files = os.listdir(path_)
        if terrain_rgb_ not in files:
            return
        if heightmap_ not in files:
            return

        self._terrain_img = Image.open(os.path.join(path_, terrain_rgb_))
        self._heightmap_img = Image.open(os.path.join(path_, heightmap_))

        self._width, self._height = self._terrain_img.size
        self._heightmap_width, self._heightmap_height = self._heightmap_img.size

        if self._width != self._heightmap_width or self._height != self._heightmap_height:
            return

        # self._geo_grid = np.zeros((self._width, self._height), dtype=geo_data_type)
        self._geo_grid = np.zeros((self._width, self._height, 5), dtype=np.uint8)
        # for dynamic object data
        self._geo_grid_dynamic = np.zeros((self._width, self._height, 3), dtype=np.uint8)
        # 지형/지물 이미지 배경 전용
        self._geo_grid_image = np.zeros([self._width, self._height, 3])

        for i in range(0, self._width):
            for j in range(0, self._height):
                grid_data = self._geo_grid[(i, j)]
                # grid_img_data = self._geo_grid_image[(i, j)]
                tr, tg, tb = self._terrain_img.getpixel((i, j))  # this does not work for bmp
                tr1 = max(0, min(tr, 1))
                tg1 = max(0, min(tg, 1))
                tb1 = max(0, min(tb, 1))
                grid_data = (tr, tg, tb, 0, 0)
                self._geo_grid_image[(i, j)] = [tr, tg, tb]

                # 아래는 height 값을 geo_grid에 입력하는 부분인데 height는 별도로 존재하기로 함. 20230826 hoyabina
                # h = self._heightmap_img.getpixel((i, j))  # this does not work for bmp
                # grid_data[3] = h[0]

                self._geo_grid[(i, j)] = grid_data

                # debug
                # if h[0] != 0 and j < 90:
                #     print(i, j, self._geo_grid[(i, j)])

    def width(self):
        return self._width

    def height(self):
        return self._height

    def copy_global_box(self):
        return self._geo_grid

    def copy_abs_box(self, agent):
        # ret = np.zeros((sizet_, sizet_), dtype=geo_data_type)
        # start_x = 0 if x_index_-width_ < 0 else x_index_ - width_
        # start_y = 0 if y_index_-height_ < 0 else y_index_-height_
        # end_x = width_ if x_index_ + width_ * 0.5 > width_ else x_index_ + width_ * 0.5
        # end_y = height_ if y_index_ + height_ * 0.5 > height_ else y_index_ + height_ * 0.5
        pos_ = agent.position
        sizet_ = agent.obs_box_size
        # observation box in accordance with obs_box_size.
        start_x = int(pos_[0] - sizet_ * 0.5)
        start_y = int(pos_[1] - sizet_ * 0.5)
        end_x = start_x + sizet_
        end_y = start_y + sizet_
        # 가장자리에 의해 잘려나가는 부분을 offset으로 하고 box를 재연산.
        startx_offset = 0 - min(start_x, 0)
        starty_offset = 0 - min(start_y, 0)
        start_x = max(start_x, 0)
        start_y = max(start_y, 0)
        endx_offset = max(0, end_x - self._width)
        endy_offset = max(0, end_y - self._width)
        end_x = min(end_x, self._width)
        end_y = min(end_y, self._width)

        # print('copy_abs_box', 'pos', int(pos_[0]), int(pos_[1]), 'startx', start_x, "starty", start_y, \
        #       'endx', end_x, "endy", end_y, 'sxOffset', startx_offset, "syOffset", starty_offset, 'exOffset',\
        #       endx_offset, "eyOffset", endy_offset, self._geo_grid[start_x:end_x, start_y:end_y].shape)
        # 지형지물+객체 정보용
        obs_list = np.ones((sizet_, sizet_, 5), dtype=np.uint16)
        obs_list[startx_offset:sizet_-endx_offset, starty_offset:sizet_-endy_offset] \
            = self._geo_grid[start_x:end_x, start_y:end_y]
        # 객체용
        obs_list_dynamic = np.ones((sizet_, sizet_, 3), dtype=np.uint16)
        obs_list_dynamic[startx_offset:sizet_-endx_offset, starty_offset:sizet_-endy_offset] \
            = self._geo_grid_dynamic[start_x:end_x, start_y:end_y]

        # todo clamp data to 0 or 1, done
        agent.observation['rgbae'] = np.clip(obs_list, 0, 1)
        agent.observation['objects'] = obs_list_dynamic
        # todo 20231227 add agent.observation['height']

        # obs for fight
        _sizet_fight = agent.view_range[0]*2;
        start_x_fight = int(pos_[0] - _sizet_fight * 0.5)
        start_y_fight = int(pos_[1] - _sizet_fight * 0.5)
        end_x_fight = start_x_fight + _sizet_fight
        end_y_fight = start_y_fight + _sizet_fight
        # 가장자리에 의해 잘려나가는 부분을 offset으로 하고 box를 재연산.
        startx_fight_offset = 0 - min(start_x_fight, 0)
        starty_fight_offset = 0 - min(start_y_fight, 0)
        start_x_fight = max(start_x_fight, 0)
        start_y_fight = max(start_y_fight, 0)
        endx_fight_offset = max(0, end_x_fight - self._width)
        endy_fight_offset = max(0, end_y_fight - self._width)
        end_x_fight = min(end_x_fight, self._width)
        end_y_fight = min(end_y_fight, self._width)

        # 지형지물+객체 정보용
        obs_list_fight = np.ones((_sizet_fight, _sizet_fight, 5), dtype=np.uint16)
        obs_list_fight[startx_fight_offset:_sizet_fight-endx_fight_offset, starty_fight_offset:_sizet_fight-endy_fight_offset] \
            = self._geo_grid[start_x_fight:end_x_fight, start_y_fight:end_y_fight]
        # 객체용
        obs_list_dynamic_fight = np.ones((_sizet_fight, _sizet_fight, 3), dtype=np.uint16)
        obs_list_dynamic_fight[startx_fight_offset:_sizet_fight-endx_fight_offset, starty_fight_offset:_sizet_fight-endy_fight_offset] \
            = self._geo_grid_dynamic[start_x_fight:end_x_fight, start_y_fight:end_y_fight]

        # todo clamp data to 0 or 1, done
        agent.observation['rgbae_fight'] = np.clip(obs_list_fight, 0, 1)
        agent.observation['objects_fight'] = obs_list_dynamic_fight
        # todo 20231227 add agent.observation['height_fight']


    def reset(self):
        self._geo_grid_dynamic = np.zeros((self._width, self._height, 3), dtype=np.uint8)

    def remove_object_agent(self, object_):
        _pos = object_.position
        grid_data = self._geo_grid[(int(_pos[0]), int(_pos[1]))]
        grid_data_list = list(grid_data)
        grid_data_list[3] = 0
        for i in range(-3, 3):
            for j in range(-3, 3):
                x = max(0, min(int(_pos[0])+i, self._width - 1))
                y = max(0, min(int(_pos[1])+j, self._width - 1))
                self._geo_grid_dynamic[x, y][0] = 0
                self._geo_grid_dynamic[x, y][1] = 0
                self._geo_grid_dynamic[x, y][2] = 0
        self._geo_grid[(int(_pos[0]), int(_pos[1]))] = tuple(grid_data_list)

    def put_object_agent(self, object_):
        _pos = object_.position
        grid_data = self._geo_grid[(int(_pos[0]), int(_pos[1]))]
        grid_data_list = list(grid_data)
        grid_data_list[3] = 1
        for i in range(-3, 3):
            for j in range(-3, 3):
                x = max(0, min(int(_pos[0])+i, self._width - 1))
                y = max(0, min(int(_pos[1])+j, self._width - 1))
                self._geo_grid_dynamic[x, y][0] = 255
                self._geo_grid_dynamic[x, y][1] = 255
                self._geo_grid_dynamic[x, y][2] = 255
                if object_.active is False:
                    self._geo_grid_dynamic[x, y][1] = 64
                    self._geo_grid_dynamic[x, y][2] = 0

        self._geo_grid[(int(_pos[0]), int(_pos[1]))] = tuple(grid_data_list)

    def remove_object_enemy(self, object_):
        _pos = object_.position
        grid_data = self._geo_grid[(int(_pos[0]), int(_pos[1]))]
        grid_data_list = list(grid_data)
        grid_data_list[4] = 0
        for i in range(-3, 3):
            for j in range(-3, 3):
                x = max(0, min(int(_pos[0])+i, self._width - 1))
                y = max(0, min(int(_pos[1])+j, self._width - 1))
                self._geo_grid_dynamic[x, y][0] = 0
                self._geo_grid_dynamic[x, y][1] = 0
                self._geo_grid_dynamic[x, y][2] = 0
        self._geo_grid[(int(_pos[0]), int(_pos[1]))] = tuple(grid_data_list)

    def put_object_enemy(self, object_):
        _pos = object_.position
        grid_data = self._geo_grid[(int(_pos[0]), int(_pos[1]))]
        grid_data_list = list(grid_data)
        grid_data_list[4] = 1
        for i in range(-3, 3):
            for j in range(-3, 3):
                x = max(0, min(int(_pos[0])+i, self._width - 1))
                y = max(0, min(int(_pos[1])+j, self._width - 1))
                self._geo_grid_dynamic[x, y][0] = 255
                self._geo_grid_dynamic[x, y][1] = 0
                self._geo_grid_dynamic[x, y][2] = 255
                if object_.active is False:
                    self._geo_grid_dynamic[x, y][1] = 64
                    self._geo_grid_dynamic[x, y][2] = 0
        self._geo_grid[(int(_pos[0]), int(_pos[1]))] = tuple(grid_data_list)

    def get_render_image(self):
        # 지형/지물 이미지 배경 전용임.
        return self._geo_grid_image

    def get_grid_RGB_property(self, x_, y_):
        if x_ < 0:
            x_ = 0
        if y_ < 0:
            y_ = 0
        if x_ >= self._width:
            x_ = self._width - 1
        if y_ >= self._height:
            y_ = self._height - 1

        tr, tg, tb = self._terrain_img.getpixel((x_, y_))
        return (tr, tg, tb)

    #20231228 add 3 callback functions for Dynamics
    def get_x_and_y_slope(self, x, y):
        return 0, 0 #self.xslope, self.yslope

    def get_z(self, x, y):
        return 0  #(0 - self.a * x - self.b * y) / self.c

    def get_friction_coefficient(self, x, y):
        return 0.8  #self.mu


