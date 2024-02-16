
## Desciption:  
This project is created for AI-CREW with LARR.

## Installation
Step 1: Download the project and cd to this project. Make sure that you have a '\runs' and '\models' folder in your projects.    
Step 2: Run the 'main.py' for training, and 'test.py' for test.    

### Checkpointing
- /saved_models: directory where model will be saved.
- /runs: directory where intermediate training results will be saved.

## Wrapper

가속화 환경 가장 바깥 부분인 envs_level0에 explore_wrapper로 wrapping
- obs, reward, done의 자료형이 MPE(Multi-Agent Particle Environment)와 유사해지도록 후처리함
-- calculate_reward, get_observation 함수 참고

- reward: 적 4개 발견 시 large bonus + 각 에이전트와 적 사이 negative l2 distance << 리워드 함수 수정해도 된다.

가속화 환경 step의 출력인 info에 각 에이전트의 yaw 값 추가
- 수정된 observation 구성: 12차원 = 2 (global x, y) + 2 (sin(yaw), cos(yaw)) + 2 (적의 상대 위치) * 4
- observation matrix: 4(아군 에이전트 수) * 12(각 에이전트의 observation) = 48차원

전차의 파라미터는 parameter.json에서 관리 가능

- lim_v_min, lim_v_max : 적용 안 되는 항목. 추후 설명할 감쇠율 통해서 간접적으로 설정 가능
- lat_flag: 1이면 횡방향 외란 적용
- dis_flag: 1이면 환경에 대한 외란 적용
- Y2: 각속도의 감쇠율 (2차식 계수)
- Y1: 각속도의 감쇠율 (1차식 계수)
- Y0: 각속도의 감쇠율 (상수항, 키우면 전차가 느려짐)
- D2: 종방향 속도의 감쇠율: (2차식 계수)
- D1: 종방향 속도의 감쇠율: (1차식 계수)
- D0: 종방향 속도의 감쇠율: (상수항, 키우면 전차가 느려짐)