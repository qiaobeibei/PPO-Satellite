import numpy as np
from gym import spaces
import satellite_function as sf
import gym
from single_pluse_model import real_time_data_process

# 下面类用来定义环境
class satellites:

    # __annotations__用于存储变量注释信息的字典
    __annotations__ = {
        "Pursuer_position": np.ndarray,
        "Pursuer_vector": np.ndarray,
        "Escaper_position": np.ndarray,
        "Escaper_vector": np.ndarray
    }

    """
    卫星博弈环境
    Arguments:
        sizes (Tuple[Tuple[int]]):
        aspect_ratios (Tuple[Tuple[float]]):
    """
    def __init__(self, Pursuer_position=np.array([2000, 2000, 1000]), Pursuer_vector=np.array([1.71, 1.14, 1.3]),
                 Escaper_position=np.array([1000, 2000, 0]), Escaper_vector=np.array([1.71, 1.14, 1.3]),
                 M=0.4, dis_safe=1000, d_capture=100000, Flag=0, fuel_c=150, fuel_t=150, d_range= 20000, args=None):

        self.Pursuer_position = Pursuer_position
        self.Pursuer_vector = Pursuer_vector
        self.Escaper_position = Escaper_position
        self.Escaper_vector = Escaper_vector
        self.White_position = Pursuer_position
        self.White_vector = Escaper_vector
        self.dis_dafe = dis_safe            # 碰撞距离
        self.d_capture = d_capture          # 抓捕距离
        self.Flag = Flag                    # 训练追捕航天器(0)、逃逸航天器(1)或者测试的标志(2)
        self.M = M                          # 航天器质量
        self.burn_reward = 0
        self.win_reward = 100
        self.dangerous_zone = 0             # 危险区数量
        self.fuel_c = fuel_c                # 抓捕航天器燃料情况
        self.fuel_t = fuel_t                # 抓捕航天器燃料情况
        self.dis = np.inf                   # 博弈距离
        self.dis_at = 0
        self.d_range = 0              # 使用可达域博弈的最小距离
        self.max_episode_steps = args.max_episode_steps
        self.ellipse_params = []            # 椭圆拟合参数
        # 椭圆拟合训练网络
        self.trian_elliptical_fitting = real_time_data_process.network_method_train(pretrain=False)
        # 下面声明其基本属性
        position_low = np.array([-500000, -500000, -500000])
        position_high = np.array([500000, 500000, 500000])
        velocity_low = np.array([-10000, -10000, -10000])
        velocity_high = np.array([10000, 10000, 10000])
        observation_low = np.concatenate((position_low, velocity_low))
        observation_high = np.concatenate((position_high, velocity_high))
        self.observation_space = spaces.Box(low=observation_low, high=observation_high, shape=(6,), dtype=np.float32)
        self.action_space = np.array([[-0.004, 0.004],
                                      [-0.004, 0.004],
                                      [-0.004, 0.004]])
        n_actions = 5  # 策略的数量
        self.action_space_beta = gym.spaces.Discrete(n_actions)    # {0,1,2,3,4}


    # 下面对Satellite的初始位置进行更新
    def reset(self, Flag):
        self.Pursuer_position = np.array([62.3367, 81.3027634, 3.36439207])
        self.Pursuer_vector = np.array([-0.0660084175, -0.0291376, 0.0147651449])
        self.pursuer_reward = 0.0
        self.Escaper_position = np.array([0, 0, 0])
        self.Escaper_vector = np.array([0.00001, 0.00001, 0.00001])
        self.escaper_reward = 0.0
        self.White_position = np.array([59.3367, 80.3027634, 4.36439207])
        self.White_vector = np.array([-0.0660084175, -0.0291376, 0.0147651449])
        self.fuel_c = 150                # 抓捕航天器燃料情况
        self.fuel_t = 150                # 抓捕航天器燃料情况
        self.origin_fuel = 150
        self.Flag = Flag

        s = np.array([self.Pursuer_position - self.Escaper_position, self.Pursuer_vector - self.Escaper_vector,
                      self.Pursuer_position, self.Pursuer_vector, self.Escaper_position, self.Escaper_vector,
                      self.Escaper_position - self.White_position, self.Escaper_vector - self.White_vector]).ravel()

        return s

    def step(self, pursuer_action, escaper_action, epsiode_count):
        DW = False
        self.pursuer_reward = 0
        self.escaper_reward = 0

        dis = np.linalg.norm(self.Pursuer_position - self.Escaper_position) # 上一状态的距离
        dis_w = np.linalg.norm(self.Escaper_position - self.White_position)
        # print(dis_w, dis, epsiode_count)
        # print(self.Pursuer_vector,self.Escaper_vector,epsiode_count)
        # print(epsiode_count)
        pursuer_action = [np.clip(action, -0.004, 0.004) for action in pursuer_action]
        escaper_action = [0,0,0]

        for i in range(3):  # update vector
            self.Pursuer_vector[i] += pursuer_action[i]
            self.Escaper_vector[i] += escaper_action[i]
        self.origin_fuel -= np.abs(pursuer_action[0]) + np.abs(pursuer_action[1]) + np.abs(pursuer_action[2])

        self.fuel_c -= (np.abs(pursuer_action[0]) + np.abs(pursuer_action[1]) + np.abs(pursuer_action[2]))
        self.fuel_t -= (np.abs(escaper_action[0]) + np.abs(escaper_action[1]) + np.abs(escaper_action[2]))

        # 轨道外推
        R0_c, V0_c = self.relative_state_to_absolute_state(self.Pursuer_position, self.Pursuer_vector)
        R0_t, V0_t = self.relative_state_to_absolute_state(self.Escaper_position, self.Escaper_vector)
        R0_w, V0_w = self.relative_state_to_absolute_state(self.White_position, self.White_vector)
        # 拉格朗日系数法
        s_1, s_2 = sf.Time_window_of_danger_zone(
            R0_c=R0_c,
            V0_c=V0_c,
            R0_t=R0_t,
            V0_t=V0_t,
            u = 3.986e5).calculation_spacecraft_status(20)
        _, s_3 = sf.Time_window_of_danger_zone(
            R0_c=R0_t,
            V0_c=V0_t,
            R0_t=R0_w,
            V0_t=V0_w,
            u = 3.986e5).calculation_spacecraft_status(20)
        R0_c, V0_c = self.absolute_state_to_relative_state(s_1[0:3], s_1[3:])
        R0_t, V0_t = self.absolute_state_to_relative_state(s_2[0:3], s_2[3:])
        R0_w, V0_w = self.absolute_state_to_relative_state(s_3[0:3], s_3[3:])

        self.Pursuer_position, self.Pursuer_vector = R0_c, V0_c
        self.Escaper_position, self.Escaper_vector = R0_t, V0_t
        self.White_position, self.White_vector = R0_w, V0_w
        self.dis = np.linalg.norm(self.Pursuer_position - self.Escaper_position)
        self.dis_at = np.linalg.norm(self.Escaper_position - self.White_position)
        self.dis_st = np.linalg.norm(self.Pursuer_position - self.White_position)
        # print(self.dis,self.dis_at,self.dis_st,epsiode_count)

        if self.dis <= self.d_capture or epsiode_count >= self.max_episode_steps or self.dis > 200 or self.dis_st > 50:
            DW = True

        # if self.dis_at < 58:
        #     self.pursuer_reward = -50
        if self.dis_st > 50:
            self.pursuer_reward = -50
        if self.dis < self.d_capture:
            self.pursuer_reward = 100

        self.pursuer_reward += 2 if self.dis < dis else -3   # 接近目标给予奖励

        pv1 = self.reward_of_action3(self.Pursuer_position)
        pv2 = self.reward_of_action1(self.Pursuer_vector)
        pv3 = self.reward_of_action2(self.Pursuer_position, self.Pursuer_vector)
        pv4 = self.reward_of_action4(self.Pursuer_position - self.Escaper_position, pursuer_action)
        # TODO PSO寻优或者加入指数项，随着距离的缩短每个方法所占的比例应该有所调整
        # TODO pv2和其他几个与位置和脉冲相比较的策略应该放在最前面，放在这里是用机动后的位置和前面的脉冲相比较
        self.pursuer_reward += 0.2 * pv1
        self.pursuer_reward += 0.1 * pv2
        self.pursuer_reward += 0.1 * pv3
        self.pursuer_reward += 2.0 * pv4

        self.escaper_reward = -self.pursuer_reward
        if self.Flag == 0:
            return np.array([self.Pursuer_position - self.Escaper_position, self.Pursuer_vector - self.Escaper_vector,
                          self.Pursuer_position, self.Pursuer_vector, self.Escaper_position, self.Escaper_vector,
                             self.Escaper_position - self.White_position, self.Escaper_vector - self.White_vector]).ravel(), self.pursuer_reward, DW
        elif self.Flag == 1:

            return np.array([self.Pursuer_position - self.Escaper_position, self.Pursuer_vector - self.Escaper_vector,
                          self.Pursuer_position, self.Pursuer_vector, self.Escaper_position, self.Escaper_vector]).ravel(), self.escaper_reward, DW
        elif self.Flag == 3:

            return np.array([self.Pursuer_position - self.Escaper_position, self.Pursuer_vector - self.Escaper_vector,
                          self.Pursuer_position, self.Pursuer_vector, self.Escaper_position, self.Escaper_vector,
                             self.Escaper_position - self.White_position, self.Escaper_vector - self.White_vector]).ravel(), self.pursuer_reward, \
                   self.fuel_c, self.fuel_t, DW, self.origin_fuel

    def calculate_number_hanger_area(self):
        # 求危险区数量需将航天器相对状态转换至惯性系下
        R0_c, V0_c = self.relative_state_to_absolute_state(self.Pursuer_position, self.Pursuer_vector)
        R0_t, V0_t = self.relative_state_to_absolute_state(self.Escaper_position, self.Escaper_vector)
        # print("R0_c, V0_c, fuel_c:", R0_c, V0_c , self.fuel_c)
        # 之所以用剩余的燃料作为最大脉冲，是因为如果在最大燃料处检测到危险区那说明刚好可以追到，但是如果在最大脉冲下也没有危险区那说明不可能有危险区
        number_zone = sf.Time_window_of_danger_zone(
            R0_c=R0_c,
            V0_c=V0_c,
            R0_t=R0_t,
            V0_t=V0_t,
            Delta_V_c=self.fuel_c,
            time_step=1
        )
        # 危险区数量
        self.dangerous_zone = number_zone.calculate_number_of_hanger_area()

    @staticmethod
    def relative_state_to_absolute_state(R0, V0):
        assert isinstance(R0, np.ndarray) and isinstance(V0, np.ndarray)
        R_cw = np.array([-8434.27083, -109.057607, 0.0369175668])
        V_cw = np.array([0.0847578275, -6.87092967, 0.000116115096])

        R_ref = R_cw + R0
        V_ref = V_cw + V0

        return R_ref, V_ref

    @staticmethod
    def absolute_state_to_relative_state(R0, V0):
        assert isinstance(R0, np.ndarray) and isinstance(V0, np.ndarray)
        R_cw = np.array([-8434.27083, -109.057607, 0.0369175668])
        V_cw = np.array([0.0847578275, -6.87092967, 0.000116115096])

        R_ref = R0 - R_cw
        V_ref = V0 - V_cw

        return R_ref, V_ref

    # 1、直追行为奖励: 以速度方向和逃逸星速度方向的相似度为奖励函数一部分
    def reward_of_action1(self, temp1):
        """
        :param temp1: 追逐星的速度向量
        :return pv1: 追逐星与逃逸星的相似度
        """
        temp1 = temp1 / np.linalg.norm(temp1)  # 单位化向量
        temp2 = self.Escaper_vector / np.linalg.norm(self.Escaper_vector)
        pv1 = np.dot(temp1, temp2)
        return pv1

    # 2、追踪行为奖励：追逐星向卫星上一时刻位置移动，形成追踪。以速度方向和距离连线方向的相似度为奖励函数一部分
    def reward_of_action2(self, temp3, temp4):
        """

        :param temp3: 追逐星的位置向量
        :param temp4: 追逐星的速度向量
        :return pv2: 追逐星速度方向与其和逃逸星上一时刻位置连线的相似度
        """
        temp3 = temp3 - self.Escaper_position
        temp3 = temp3 / np.linalg.norm(temp3)  # 单位化向量
        temp4 = temp4 / np.linalg.norm(temp4)  # 单位化向量
        pv2 = np.dot(temp3, temp4)
        return pv2

    def reward_of_action3(self, temp1):
        """

        :param temp1: 追逐星的位置向量
        :return pv3: 追逐星速度方向与其和逃逸星上一时刻位置连线的相似度
        """
        temp1 = temp1 / np.linalg.norm(temp1)  # 单位化向量
        temp2 = self.Escaper_position / np.linalg.norm(self.Escaper_position)
        pv3 = np.dot(temp1, temp2)
        return pv3

    # 两航天器状态的差值与脉冲的负相似度
    def reward_of_action4(self, temp1, temp2):
        """
        :param temp1: 两航天器位置的差值
        :param temp2: 脉冲
        :return pv4: 两航天器位置的差值与脉冲的负相似度
        """
        if temp2[0] != 0 and temp2[1] != 0 and temp2[2] != 0:
            temp1 = temp1 / np.linalg.norm(temp1)  # 单位化向量
            temp2 = np.array(temp2)
            temp2 = temp2 / np.linalg.norm(temp2)
            pv4 = -np.dot(temp1, temp2)
        else:
            pv4 = 0

        return pv4

    # 航天器速度差值与脉冲的相似度(为了保持一定的距离不被甩开)
    def reward_of_action5(self, temp1, temp2):
        """
        :param temp1: 两航天器速度的差值
        :param temp2: 脉冲
        :return pv4: 航天器速度差值与脉冲的相似度
        """
        if temp2[0] != 0 and temp2[1] != 0 and temp2[2] != 0:
            temp1 = temp1 / np.linalg.norm(temp1)  # 单位化向量
            temp2 = np.array(temp2)
            temp2 = temp2 / np.linalg.norm(temp2)
            pv5 = -np.dot(temp1, temp2)
        else:
            pv5 = 0

        return pv5
