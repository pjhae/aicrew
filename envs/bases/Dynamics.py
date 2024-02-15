import numpy as np
import math
# from Env import Env
from scipy.signal import cont2discrete

#==================================================================================================
class Dynamics:
    ##########################################################################################################################################
    ####################### ENVIRONMENT BLOCK ################################################################################################
    ##########################################################################################################################################
    def get_x_and_y_slope(self):
        x = self.x
        y = self.y
        
        xslope, yslope = self.env.get_x_and_y_slope(x, y) # environment api function
        
        return xslope, yslope
        
    def get_z(self):
        x = self.x
        y = self.y
        
        z = self.env.get_z(x, y) # environment api function
        
        return z
    
    def get_friction_coefficient(self):
        x = self.x
        y = self.y
        
        mu = self.env.get_friction_coefficient(x, y) # environment api function
        
        return mu
    ##########################################################################################################################################
    
    
    #==============================================================================================
    def plant_lat(self):
        vy = self.vy
        ay = self.ay
        dt = self.dt
        
        next_vy = vy * 0.9 + ay * dt
        next_vy = math.trunc(next_vy * 1000)/1000
        
        return next_vy
    
    #==============================================================================================
    def plant(self, a_des, ar_des, a_dis, ar_dis, a_friction, ar_friction):
        # Dynamics
        # X = [v, yawrate, a, ar]'
        v = self.v
        yawrate = self.yawrate
        acc = self.acc
        ar = self.ar
        Ad = self.Ad
        Bd = self.Bd
        dt = self.dt
        
        X = np.array([acc, ar]).T
        U = np.array([a_des, ar_des]).T
        A1 = np.matmul(Ad, X) + np.matmul(Bd, U)
        
        A2 = np.array([a_dis, ar_dis]).T + np.array([a_friction, ar_friction]).T
        A = A1 + A2

        pre_v = v
        pre_yawrate = yawrate

        # discrete update
        v = v + A[0] * dt
        yawrate = yawrate + A[1] * dt # yawrate decay

        
        # Disturbance cannot make movement reverse!
        # 부호가 바뀜
        fric_a_sign = 0
        if a_friction != 0:
            fric_a_sign = a_friction/abs(a_friction)
        
        fric_yawrate_sign = 0
        if ar_friction != 0:
            fric_yawrate_sign = ar_friction/abs(ar_friction)

        pre_v_sign = 0
        if pre_v != 0:
            pre_v_sign = pre_v / abs(pre_v)

        v_sign = 0
        if v != 0:
            v_sign = v / abs(v)

        pre_yawrate_sign = 0
        if pre_yawrate != 0:
            pre_yawrate_sign = pre_yawrate / abs(pre_yawrate)

        yawrate_sign = 0
        if yawrate != 0:
            yawrate_sign = yawrate / abs(yawrate)

        ###############
        # 부호가 바뀜 #
        ###############
        # v
        if pre_v_sign * v_sign == -1:
            v = 0

            # 부호가 바뀐것에 영향이 friction
            if v_sign == fric_a_sign:
                A2[0] = -1 * A1[0]

        # yawrate
        if pre_yawrate_sign * yawrate_sign == -1:
            yawrate = 0

            # 부호가 바뀐것에 영향이 friction
            if yawrate_sign == fric_yawrate_sign:
                A2[1] = -1 * A1[1]
        
        
        X_next = np.array([v, yawrate, A1[0], A1[1]]).T
        
        return X_next
    
    
    
    #==============================================================================================
    def __init__(self, env, dt, ini_x, ini_y, ini_v, ini_yaw, ini_yawrate, ini_a, ini_ar, config): 
        self.env = env # 환경 접근자
        self.init_parameter(dt, ini_x, ini_y, ini_v, ini_yaw, ini_yawrate, ini_a, ini_ar, config)


    #==============================================================================================
    def init_parameter(self, dt, ini_x, ini_y, ini_v, ini_yaw, ini_yawrate, ini_a, ini_ar, config):
        # json 파일을 이용해 parameters 관리
        # dt        = 0.01 [sec] : sampling time                                       -> 가속화 요구사항과 동역학 안정성을 고려해야 최적값을 찾을 예정
        # tau_ax    = 1.0  [-]   : time delay coefficient of longitudinal acceleration -> 종방향 가속도 차량 동특성을 반영하게 될 튜닝 계수
        # tau_alpha = 1.0  [-]   : time delay coefficient of angular acceleration      -> 회전   가속도 차량 동특성을 반영하게 될 튜닝 계수
        # ini_x     = X    [m]   : initial X-position in global coordinate (visual)    -> 영상엔진에서 전역좌표계 초기위치 X 
        # ini_y     = Y    [m]   : initial Y-position in global coordinate (visual)    -> 영상엔진에서 전역좌표계 초기위치 Y
        # ini_yaw   = yaw  [deg] : initial yaw angle in global coordinate (visual)     -> 영상엔진에서 전역좌표계 초기회전각도 yaw 
    
        
        ################
        # Vehicle info #
        ################
        self.MT = config["MT"] # vehicle mass
        self.g = config["g"] # gravity acceleration
        self.HM = config["HM"] # center of mass height from ground
        self.W = config["W"] # vehicle width
        self.l1 = config["l1"] # vehicle length between front wheel and center of mass
        self.l2 = config["l2"] # vehicle length between rear wheel and center of mass
        self.mu = config["mu"] # ground frictio coefficient
        self.Iz = config["Iz"] # z-axis moment of inertia

        self.tau_a = config["tau_a"] # acceleration time delay
        self.tau_r = config["tau_r"] # angular acceleration time delay
        
        self.dt = dt # time interval
        self.Ad, self.Bd = self.get_matrix() # dynamics matrix
        
        self.D2 = config["D2"] # Power loss coefficient
        self.D1 = config["D1"] # Power loss coefficient
        self.D0 = config["D0"] # Power loss coefficient
        self.Y2 = config["Y2"] # yaw rate friction coefficient
        self.Y1 = config["Y1"] # yaw rate friction coefficient
        self.Y0 = config["Y0"] # yaw rate friction coefficient
        
        self.rho = config["rho"] # air density
        
        self.Bx = np.array([1, 0, 0]).T
        self.By = np.array([0, 1, 0]).T
        self.Bz = np.array([0 ,0, 1]).T
        
        self.delta = None # XYZ -> XY
         
        self.MG = np.array([0, 0, -self.MT*self.g]).T # gravity
        self.Decomposed = None
        
        self.a = None # left rear Wheel's Normal force
        self.b = None # left front Wheel's Normal force
        self.c = None # right rear Wheel's Normal force
        self.d = None # right front Wheel's Normal force
        
        
        #################
        # Vehicle State #
        ################# 
        self.xslope = 0 # global x axis slope
        self.yslope = 0 # global y axis slope
        
        self.x = ini_x # vehicle x position
        self.y = ini_y # vehicle y position
        self.z = None  # vehicle z position // get from env
        
        self.yaw = ini_yaw # vehicle yaw
        self.v = ini_v # vehicle velocity
        self.yawrate = ini_yawrate # yawrate 
        self.acc = ini_a # acceleration
        self.ar = ini_ar # angular acceleration
        
        self.vy = 0 # vehicle lateral velocity
        self.ay = 0 # vehicle lateral acceleration
        
        ##########################
        # Dynamics module on/off #
        ##########################
        self.lat_flag = config["lat_flag"] # lateral movement on/off
        self.dis_flag = config["dis_flag"] # disturbance on/off
        
        
        ####################
        # Environment info #
        ####################
        self.Lx = None # vehicle's x axis gravity component
        self.Ly = None # vehicle's y axis gravity component
        self.Lz = None # vehicle's z axis gravity component
        
        self.Rd = 1/2*self.rho*self.v*self.v # aerial drag force
        self.v_sign = 0
        if self.v != 0:
            self.v_sign = self.v/abs(self.v)
        self.Rd = self.v_sign * self.Rd
        
        self.z = self.get_z() # gorund height
        self.x_slope, self.y_slope = self.get_x_and_y_slope() # ground slope
        self.mu = self.get_friction_coefficient() # ground friction coefficient
        self.Bx, self.By, self.Bz = self.get_basis_vector() # Tracked Vehicle local coordinate's basis vector
        self.delta = self.get_tracked_vehicle_slope()
        self.Lx, self.Ly, self.Lz = self.gravity_decompose() # Gravity decomposed by local coordinate
    

    #==============================================================================================
    def get_matrix(self):
        # 다이나믹스에 사용되는 시스템 matrix(Ad), 입력 matrix(Bd)을 return 하는 함수
        tau_a = self.tau_a
        tau_r = self.tau_r
        dt = self.dt
        
        A = np.array([[-1/tau_a, 0], [0, -1/tau_r]])
        B = np.array([[1/tau_a, 0], [0, 1/tau_r]])
        
        C = np.zeros((2,2))
        D = np.zeros((2,2))

        Ad, Bd, Cd, Dd, _ = cont2discrete((A, B, C, D), dt, method='zoh')
        
        return Ad, Bd
    
    
    #==============================================================================================
    def get_basis_vector(self):
        # 전차 좌표계를 얻는 과정
        xslope = self.x_slope
        yslope = self.y_slope
        yaw = self.yaw
        
        a = -math.tan(yslope)
        b = -math.tan(xslope)
        c = 1

        Normal = np.array([a,b,c]) / math.sqrt(a**2+b**2+1)

        r = [math.tan(xslope), math.tan(yslope), 0]

        alpha = -math.atan(math.sqrt(math.tan(xslope)**2+math.tan(yslope)**2))
        if yslope != 0:
            beta = -math.atan(math.tan(xslope)/math.tan(yslope)) # singular point ex) A/0
        else: # yslope == 0
            beta = -1.5708 # radian
            
            
        #################
        # Rotate Matrix #
        #################
        R1=np.array([[math.cos(beta), -math.sin(beta), 0], [math.sin(beta), math.cos(beta), 0], [0, 0, 1]])
        
        R2=np.array([[math.cos(alpha), 0, math.sin(alpha)], [0, 1, 0], [-math.sin(alpha), 0, math.cos(alpha)]])
        
        R3=np.array([[math.cos(beta), math.sin(beta), 0], [-math.sin(beta), math.cos(beta), 0], [0, 0, 1]])
        
        R = np.dot(R3, np.dot(R2,R1))

        if (xslope == 0) and (yslope == 0):
            R = np.diag(np.array([1, 1, 1]))

        Rz = np.array([[math.cos(yaw), -math.sin(yaw), 0], [math.sin(yaw), math.cos(yaw), 0], [0, 0, 1]])
        
        x = np.array([1,0,0]).T
        y = np.array([0,1,0]).T
        z = np.array([0,0,1]).T
        
        lx = np.dot(Rz, x)
        ly = np.dot(Rz, y)
        lz = np.dot(Rz, z)

        Bx = np.dot(R, lx)
        By = np.dot(R, ly)
        Bz = np.dot(R, lz)
        
        
        ##############################################################
        # tank front x in dynacmis
        # tank front y in tmps 
        temp_ = Bx
        Bx = By
        By = temp_
        ##############################################################3
        
        return Bx, By, Bz 
    
    
    #==============================================================================================
    def get_tracked_vehicle_slope(self):
        Bx = self.Bx
        
        delta = 0
        
        if math.sqrt(Bx[0]**2 + Bx[1]**2) == 0:
            delta = math.pi/2
        else:
            delta = math.atan(Bx[2]/math.sqrt(Bx[0]**2 + Bx[1]**2))
            
        return delta
    
    
    #==============================================================================================
    def gravity_decompose(self):
        Bx = self.Bx
        By = self.By
        Bz = self.Bz
        MG = self.MG
        
        A = np.column_stack((Bx, By))
        A = np.column_stack((A, Bz))
        Decomposed = np.linalg.solve(A, MG)
        
        Lx = Decomposed[0]
        Ly = Decomposed[1]
        Lz = Decomposed[2]
        
        return Lx, Ly, Lz
        
    
    #==============================================================================================
    def get_Normal_Force(self):
        Lx = self.Lx
        Ly = self.Ly
        Lz = self.Lz
        
        MT = self.MT
        HM = self.HM
        W = self.W
        l1 = self.l1
        l2 = self.l2
        
        ay = self.ay
        ax = self.acc
        
        Rd = self.Rd

        a = -Lz * ((HM*Ly/W)+(MT*ay/W) - Lz/2)/(-Lz) * ( (-l1*Lz) - HM*Lx+ HM*(MT * ax + Rd))/(l2 + l1)/(-Lz)
        if a < 0:
            a = 0
    
        b = (HM*Ly/W+(MT*ay/W)) - Lz/2 - a
        if b < 0:
            b = 0
        
        c = ((-l1*Lz) - HM*Lx + HM*(MT*ax + Rd))/(l2 + l1) - a
        if c < 0:
            c = 0
        
        d = -Lz - a - b - c
        if d < 0:
            d = 0
        
        return a,b,c,d
        
        
    #==============================================================================================    
    def get_disturbance(self):
        ax = 0
        bx = 0
        cx = 0
        dx = 0
        ay = 0
        by = 0
        cy = 0
        dy = 0
        
        lat_accel = 0
        a = self.a
        b = self.b
        c = self.c
        d = self.d
        
        v = self.v

        Lx = self.Lx
        Ly = self.Ly
        Lz = self.Lz
        
        MT = self.MT
        W = self.W
        l2 = self.l2
        l1 = self.l1
        
        Iz = self.Iz
        mu = self.mu
        
        Rd = self.Rd
    
        # 방향은 N1 -> N2(커지는 방향), V의 방향의 반대
        alpha_sign = 1
        if (a + b) < (c + d):
            alpha_sign = -1
        

        # longitudinal
        long_sign = 0

        V_sign = 1
        if v == 0:
            if Lx != 0:
                long_sign = -1 * Lx/abs(Lx)

        if v != 0:
            V_sign = v/abs(v)
            long_sign = -1 * v/abs(v)

        ax = long_sign * mu*a
        bx = long_sign * mu*b
        cx = long_sign * mu*c
        dx = long_sign * mu*d



        if abs(ax+bx+cx+dx) > abs(Lx):
            aax = ax * abs(Lx/abs(ax+bx+cx+dx))
            bbx = bx * abs(Lx/abs(ax+bx+cx+dx))
            ccx = cx * abs(Lx/abs(ax+bx+cx+dx))
            ddx = dx * abs(Lx/abs(ax+bx+cx+dx))

            ax = aax
            bx = bbx
            cx = ccx
            dx = ddx


        # lateral
        lat_sign = 0
        if Ly != 0:
            lat_sign = -1 * Ly/abs(Ly)

        ay = long_sign * math.sqrt( (mu*a)**2 - ax**2)
        by = long_sign * math.sqrt( (mu*b)**2 - bx**2)
        cy = long_sign * math.sqrt( (mu*c)**2 - cx**2)
        dy = long_sign * math.sqrt( (mu*d)**2 - dx**2)
        
        if abs(Ly) < abs(ay+by+cy+dy):
            aay = lat_sign * abs(Ly/(ay+by+cy+dy)*ay)
            bby = lat_sign * abs(Ly/(ay+by+cy+dy)*by)
            ccy = lat_sign * abs(Ly/(ay+by+cy+dy)*cy)
            ddy = lat_sign * abs(Ly/(ay+by+cy+dy)*dy)

            ay = aay
            by = bby
            cy = ccy
            dy = ddy
        else:
            Ly_sign = 0
            
            if Ly != 0:
                Ly_sign = Ly/abs(Ly)
            
            lat_accel = Ly_sign * (abs(Ly) - abs(ay +by + cy + dy))/MT
        

        ################
        # acceleration #
        ################
        a = (Lx + (ax+bx+cx+dx) - V_sign * Rd)/MT; 
        a = math.trunc(10000*a)/10000


        ########################
        # angular acceleration #
        ########################
        # 방향은 N1 -> N2(커지는 방향), V의 방향의 반대
        # alpha = -1 * V_sign * (-W/2*(cx+dx) + W/2*(ax+bx) - l2*(ay+cy) + l1*(by+dy))/Iz;
        alpha = abs(-W/2*(cx+dx) + W/2*(ax+bx) - l2*(ay+cy) + l1*(by+dy))/Iz
        alpha = V_sign * alpha * alpha_sign

        alpha_sign = 0
        if alpha != 0:
            alpha_sign = alpha/abs(alpha)
            
        alpha = alpha_sign * math.trunc(10000*abs(alpha))/10000
    
        return a, alpha, lat_accel
   
    #==============================================================================================
    def get_friction(self):
        Lx = self.Lx
        
        v = self.v
        yawrate = self.yawrate
        
        D2 = self.D2
        D1 = self.D1
        D0 = self.D0
        Y2 = self.Y2
        Y1 = self.Y1
        Y0 = self.Y0
        
        MT = self.MT
        
        long_sign = 0
        V_sign = 0
        if v == 0:
            if Lx != 0:
                long_sign = -1 * Lx/abs(Lx)

        if v != 0:
            V_sign = v/abs(v)
            long_sign = -1 * v/abs(v)

        ##################
        # longi friction #
        ##################
        Fd = long_sign * abs(D2*v**2) + long_sign * abs(D1 * v)
        if v != 0:
            Fd = Fd + long_sign * D0
        
        a_fric = (- V_sign * abs(Fd))/MT; 


        ####################
        # yawrate friction #
        ####################
        yawrate_sign = 1
        if yawrate != 0:
            yawrate_sign = yawrate/abs(yawrate)
            
        alpha_fric = - yawrate_sign * Y2 * abs(yawrate**2)  - yawrate_sign * Y1 * abs(yawrate)

        if yawrate != 0:
            alpha_fric = alpha_fric - yawrate_sign * Y0
        
        return a_fric, alpha_fric
   
    #==============================================================================================
    def update(self, collision_flag, x, y, z): 
        # Agent의 상태를 업데이트 한다.
        # 환경에서 보낸 충돌 여부를 검사한다.
        
        if collision_flag:
            self.v = 0
            self.vy = 0
            self.yawrate = 0
            self.ay = 0
            self.acc = 0
            self.ar = 0
            
        self.x = x
        self.y = y
        
        self.z = self.get_z() # gorund height
        self.x_slope, self.y_slope = self.get_x_and_y_slope() # ground slope
        self.mu = self.get_friction_coefficient() # ground friction coefficient
   
   
    #==============================================================================================
    def step(self, ax_des, ar_des): # all float type
        self.z = self.get_z() # gorund height
        self.x_slope, self.y_slope = self.get_x_and_y_slope() # ground slope
        self.mu = self.get_friction_coefficient() # ground friction coefficient
        self.Bx, self.By, self.Bz = self.get_basis_vector() # Tracked Vehicle local coordinate's basis vector
        self.delta = self.get_tracked_vehicle_slope()
        self.Lx, self.Ly, self.Lz = self.gravity_decompose() # Gravity decomposed by local coordinate
        
        self.a, self.b, self.c, self.d = self.get_Normal_Force() # Vehicle's each wheel's normal force
        
        
        ###############
        # Disturbance #
        ###############
        self.a_dis = 0
        self.ar_dis = 0
        self.ay = 0
        if self.dis_flag:
            a_dis, ar_dis, lat_accel = self.get_disturbance() # vehicle's disturbance
            
            self.a_dis = math.trunc(a_dis*10000)/10000
            self.ar_dis = math.trunc(ar_dis*10000)/10000
            self.ay = math.trunc(lat_accel*10000)/10000
    
        if not self.lat_flag:
            self.ay = 0
            

        ############
        # Friction #
        ############
        a_fric, ar_fric = self.get_friction()
        self.a_fric = math.trunc(a_fric*10000)/10000
        self.ar_fric = math.trunc(ar_fric*10000)/10000

    

        #################
        # Next Position #
        #################
        Vxy = self.v*math.cos(self.delta)
        yawyaw = math.atan2(self.Bx[1], self.Bx[0])
        
        Vx = Vxy*math.cos(yawyaw)
        Vy = Vxy*math.sin(yawyaw)

        self.x = self.x + Vx * self.dt
        self.y = self.y + Vy * self.dt
        self.yaw = self.yaw + self.yawrate * self.dt
        
        ##################### pjhae #####################
        self.yawrate = np.clip(self.yawrate, -0.15, 0.15)
        # print(self.yawrate)
        # print("#")
        #################################################


        # lateral movement
        vyx = self.By[0] * self.vy
        vyy = self.By[1] * self.vy
        self.x = self.x + vyx * self.dt
        self.y = self.y + vyy * self.dt
        
        
        ##########
        # Update #
        ##########
        X_next = self.plant(ax_des, ar_des, self.a_dis, self.ar_dis, self.a_fric, self.ar_fric)
        
        self.v = math.trunc(X_next[0]*10000)/10000
        self.yawrate = math.trunc(X_next[1]*10000)/10000
        self.acc = math.trunc(X_next[2]*10000)/10000
        self.ar = math.trunc(X_next[3]*10000)/10000
        
        self.Rd = 1/2*self.rho*self.v*self.v
        self.v_sign = 0
        if self.v != 0:
            self.v_sign = self.v/abs(self.v)
        self.Rd = self.v_sign * self.Rd
        
        
        self.vy = self.plant_lat() 
        self.vy = math.trunc(self.vy*10000)/10000
        
        
        while(True):
            if self.yaw > 3.141592*2: # over 2pi
                self.yaw = self.yaw - 3.141592*2
            elif self.yaw < - 3.141592*2: # lower  -2pi
                self.yaw = self.yaw + 3.141592*2
            else:
                break
            
        
        return self.x, self.y, self.yaw, self.v, self.yawrate, (a_dis + a_fric), (ar_dis + ar_fric) # 수정
#==================================================================================================


#
# if __name__ == "__main__":
#     #==================================================================================================
#     import json
#     def load_config(filepath):
#         with open(filepath, 'r') as file:
#             config = json.load(file)
#         return config
#
#     config = load_config('./parameters.json') # 전차 입력 파라메터 json 파일 로드
#
#     #==================================================================================================
#     dt = 0.1
#     ini_x = 0
#     ini_y = 0
#     ini_v = 0
#     ini_yaw = 0
#     ini_yawrate = 0
#     ini_a = 0
#     ini_ar = 0
#
#     import matplotlib.pyplot as plt
#
#     # 수정: custom layout을 위해 추가
#     import matplotlib.gridspec as gridspec
#
#     from matplotlib import cm
#     from matplotlib.ticker import LinearLocator
#
#     # Simulation
#     xslope = 30/180 * math.pi
#     yslope = 30/180 * math.pi
#     veh_v = ini_v
#     veh_yaw = ini_yaw
#
#     # plane
#     a = -math.tan(yslope)
#     b = -math.tan(xslope)
#     c = 1
#
#
#     # simulation test
#     # 수정 x_min, x_max, y_min, y_max를 수정해서 경사면을 늘림.
#     x_min = -200
#     x_max = 200
#     y_min = -400
#     y_max = 200
#     dx = 5
#     dy = 5
#
#     Xs = np.arange(x_min, x_max, dx)
#     Ys = np.arange(y_min, y_max, dy)
#
#
#     # APF map
#     X = np.arange(x_min, x_max, dx)
#     Y = np.arange(y_min, y_max, dy)
#     Xs = []
#     Ys = []
#     for i in range(0, len(X)):
#         Xs.append(X[i] * np.ones_like(Y))
#         Ys.append(Y)
#
#     Xs = np.array(Xs)
#     Ys = np.array(Ys)
#     Zs = np.zeros_like(Xs)
#
#     N = np.array([a,b,c]) / math.sqrt(a**2+b**2+1)
#
#     d = 0
#     Zs = (-N[0]*Xs - N[1]*Ys - d)/N[2]
#
#
#     fig = plt.figure(figsize =(20, 10))
#     # 수정: custom layout을 위해 gs와 ax3d
#     gs = gridspec.GridSpec(5, 2, width_ratios=[2, 1])  # 3 rows, 2 columns
#     ax3d = plt.subplot(gs[:, 0], projection='3d')
#     ax1 = plt.subplot(gs[0, 1])  #Subplots for the other graphs
#     ax2 = plt.subplot(gs[1, 1])
#     ax3 = plt.subplot(gs[2, 1])
#     ax4 = plt.subplot(gs[3, 1])
#     ax5 = plt.subplot(gs[4, 1])
#
#     # 수정 inclined plane을 그리기 위해 grid 추가 & Calculate Z for the inclined plane
#     x_range = np.linspace(x_min, x_max, num=int((x_max-x_min)/dx))
#     y_range = np.linspace(y_min, y_max, num=int((y_max-y_min)/dy))
#     X_grid, Y_grid = np.meshgrid(x_range, y_range)
#     Z_grid = X_grid * math.tan(xslope) + Y_grid * math.tan(yslope)
#     ax3d.plot_surface(X_grid, Y_grid, Z_grid, color='saddlebrown', alpha=0.6)
#
#
#     env = Env(xslope, yslope)
#     vehicle = Dynamics(env, dt, ini_x, ini_y, ini_v, ini_yaw, ini_yawrate, ini_a, ini_ar, config)
#
#
#     Vs = []
#     Vys = []
#     Rs = []
#     yaws = []
#     Traj = 600
#     yawyaws = []
#     As = []
#     Ays = []
#     yaw_accels = []
#
#     a_dis = []
#     alpha_dis = []
#
#     N1s = []
#     N2s = []
#     Lxs = []
#     Lys = []
#
#     for i in range(1, Traj):
#         if i == 1:
#             ax3d.scatter(vehicle.x, vehicle.y, vehicle.get_z(), s=100, c='magenta', marker='o')
#             ax3d.plot([0, 50 * vehicle.Bx[0]],[0, 50 * vehicle.Bx[1]],[0, 50*vehicle.Bx[2]], 'black', linewidth=3, zorder = 1)
#             ax3d.plot([vehicle.x, vehicle.x + math.sqrt(vehicle.v)*vehicle.Bx[0]], [vehicle.y, vehicle.y+math.sqrt(vehicle.v)*vehicle.Bx[1]],
#                     [vehicle.z,vehicle.z + math.sqrt(vehicle.v) * vehicle.Bx[2]], 'magenta', linewidth=3, zorder = 1)
#             ax3d.plot([vehicle.x, vehicle.x + vehicle.Bx[0]], [vehicle.y, vehicle.y + vehicle.Bx[1]],
#                     [vehicle.z,vehicle.z+vehicle.Bx[2]], 'red', linewidth=3, zorder = 1)
#
#         vehicle.step(0, 0)
#         z = env.get_z(vehicle.x, vehicle.y)
#
#         if i%3 == 0:
#             ax3d.plot([vehicle.x, vehicle.x+vehicle.Bx[0]], [vehicle.y, vehicle.y+vehicle.Bx[1]],
#                     [vehicle.z, vehicle.z+vehicle.Bx[2]], 'r-', linewidth=3, zorder = 1)
#             ax3d.plot([vehicle.x, vehicle.x+vehicle.By[0]], [vehicle.y, vehicle.y+vehicle.By[1]],
#                     [vehicle.z, vehicle.z+vehicle.By[2]], 'g-', linewidth=3, zorder = 1)
#             ax3d.plot([vehicle.x, vehicle.x+vehicle.Bz[0]], [vehicle.y, vehicle.y+vehicle.Bz[1]],
#                     [vehicle.z, vehicle.z+vehicle.Bz[2]], 'b-', linewidth=3, zorder = 1)
#
#             # # force
#             # ax.plot3([x, x+Lx/5e4*Bx(1)], [y, y+Lx/5e4*Bx(2)], [z,z+Lx/5e4*Bx(3)], 'r--', 'LineWidth',3);
#             # ax.plot3([x, x+Ly/5e4*By(1)], [y, y+Ly/5e4*By(2)], [z,z+Ly/5e4*By(3)], 'g--', 'LineWidth',3);
#             # ax.plot3([x, x+Lz/5e4*Bz(1)], [y, y+Lz/5e4*Bz(2)], [z,z+Lz/5e4*Bz(3)], 'b--', 'LineWidth',3);
#
#             # % direction
#             # plot3([x, x+sqrt(V)*Bx(1)], [y, y+sqrt(V)*Bx(2)], [z,z+sqrt(V)*Bx(3)], 'm-', 'LineWidth',3);
#
#             ax3d.scatter(vehicle.x, vehicle.y, vehicle.get_z(), s=100, c='k', marker='o')
#
#         # 수정: Vs, Rs, As, yaw_accels, yaws, yawyaws를 subplot으로 그리기 위해 추가
#         print(f"vechicle.v: {vehicle.v}")
#         Vs.append(vehicle.v)
#         Vys.append(vehicle.vy)
#         Rs.append(vehicle.yawrate)
#         As.append(vehicle.acc)
#         Ays.append(vehicle.ay)
#         yaw_accels.append(vehicle.ar)
#         yaws.append(vehicle.yaw*180/math.pi)
#         #yawyaws = yawyaws.append(vehicle.yawyaw * 180/math.pi)
#
#         # a_dis.append()
#         # a_dis = [a_dis, ax];
#         # alpha_dis = [alpha_dis, alpha];
#
#         print("\n##########################################################")
#         print("vehicle_state_x: ", vehicle.x)
#         print("vehicle_state_y: ", vehicle.y)
#
#
#     ax3d.set_aspect('equal', adjustable='box')
#     ax3d.set_xlabel('X')
#     ax3d.set_ylabel('Y')
#
#     # 수정: subplot 그리기
#     ax1.plot(Vs)
#     ax1.plot(Vys)
#     ax1.set_title('Velocity')
#
#     ax2.plot(As)
#     ax2.plot(Ays)
#     ax2.set_title('Acceleration')
#
#     ax3.plot(Rs)
#     ax3.set_title('Yawrate')
#
#     ax4.plot(yaw_accels)
#     ax4.set_title('Angular Acceleration')
#
#     ax5.plot(yaws)
#     ax5.set_title('Yaw')
#
#     plt.tight_layout()
#     plt.show()
    