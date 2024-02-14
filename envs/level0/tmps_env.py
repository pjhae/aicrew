from envs.bases.geo_base import GeoGridData
from envs.bases.env_base import tmps_env_base
import envs.bases.scenario_base
import os
import envs.level0
import json

_path = os.path.dirname(os.path.abspath(__file__))
print(_path)

# json_data = {}


# def read_config_from_json():
#     with open('parameters.json', 'r') as f:
#         json_data = json.load(f)
#     print(json.dumps(json_data))

def load_config(filepath):
    with open(filepath, 'r') as file:
        config = json.load(file)
    return config

    # config = load_config('./parameters.json') # 전차 입력 파라메터 json 파일 로드

class env_level0(tmps_env_base):

    def __init__(self, configs):
        print('tmps_env __init__')
        # 20231228 read config from json file
        # read_config_from_json()
        print(f"현재 작업 디렉토리: {os.getcwd()}")
        dynamic_parameter = load_config('envs/models/parameters.json') # json_data
        
        # read geo metadata and make geo config. should be done first.
        geo_grid_data = GeoGridData(_path, '300x300.bmp', '300x300_4m.png')
        configs['map_width'] = geo_grid_data.width()
        configs['map_height'] = geo_grid_data.height()
        configs['geo_grid_data'] = geo_grid_data

        # make agent config
        agents = envs.level0.make_agents(configs['num_agents'], configs['map_width'], configs['obs_box_size'],\
                                         configs['init_pos'], configs['dynamic_delta_t'],\
                                         dynamic_parameter, geo_grid_data)
        configs['agents'] = agents

        # make enemy config
        enemies = envs.bases.scenario_base.make_enemies()
        configs['enemies'] = enemies

        super().__init__(configs)
        print(configs)

    def reset(self, **kwargs):
        print(kwargs)
        tmps_env_base.reset(self, **kwargs)

    def step(self, actions):
        # print(actions)
        return tmps_env_base.step(self, actions)


        # for agent in self.agents:
        #     agent.action = actions[num]
        #     num += 1
        #     print(agent.action)



