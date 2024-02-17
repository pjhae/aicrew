import os
import json
import shutil

def logger(arglist):
    
    # 가속화 환경 파라미터 저장
    current_dir = os.getcwd()
    params_src = 'envs/models/parameters.json'
    json_dir = os.path.join(current_dir, arglist.log_dir + '/json')
    os.makedirs(json_dir, exist_ok=True)
    
    params_src = os.path.join(current_dir, params_src)
    shutil.copy(params_src, json_dir)
    
    # 아규먼트들 저장
    args_dict = {k: safe_serialize(v) for k, v in vars(arglist).items()}
    args_file = os.path.join(json_dir, 'args.json')
    
    with open(args_file, 'w') as f:
        json.dump(args_dict, f, indent=4)
        
def safe_serialize(obj):
    if isinstance(obj, (bool, int, float, str, bytes, list, dict, tuple, type(None))):
        return obj
    return str(obj) 