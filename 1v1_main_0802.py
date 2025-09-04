# --coding:utf-8--
import matplotlib.pyplot as plt
import numpy as np
from normalization import Normalization, RewardScaling
from replaybuffer import ReplayBuffer
from ppo_continuous import PPO_continuous
from environment_0802 import satellites
from tqdm import tqdm
import plot_function as pf
import os
import pandas as pd
from datetime import datetime

def process_data(data):
    d_capture = 5000
    df = pd.DataFrame(data)
    df_sorted = df.sort_values(by="end_distances").reset_index(drop=True)
    index = df_sorted[df_sorted["end_distances"] < d_capture].index[-1]
    mean_values_before = df_sorted.iloc[:index + 1].mean()
    mean_values_after = df_sorted.iloc[index + 1:].mean()
    mean_df = pd.DataFrame({
        'start_distances_mean': [mean_values_before['start_distances'], mean_values_after['start_distances']],
        'end_distances_mean': [mean_values_before['end_distances'], mean_values_after['end_distances']],
        'pursuer_oil_mean': [mean_values_before['pursuer_oil'], mean_values_after['pursuer_oil']],
        'escaper_oil_mean': [mean_values_before['escaper_oil'], mean_values_after['escaper_oil']],
        'Round_end_mean': [mean_values_before['Round_end'], mean_values_after['Round_end']],
        'total_saved_ratio(%)': [mean_values_before['Total_saved_fuel'], mean_values_after['Total_saved_fuel']]
    })

    mean_df["Mean_Type"] = ["Mean_Before_Capture", "Mean_After_Capture"]
    output_path_means = 'means_separated.xlsx'
    file_name = datetime.now().strftime('%Y-%m-%d')
    current_date = f"{file_name}_{group_number}"
    save_dir = os.path.join("./Results",current_date)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    output_path_means = os.path.join(save_dir, 'means_separated.xlsx')
    mean_df.to_excel(output_path_means, index=False)

# 先定义一个参数类，用来储存超参数
class args_param(object):
    def __init__(self, max_train_steps=int(3e6),
                 evaluate_freq=5e3,
                 save_freq=20,
                 policy_dist="Gaussian",
                 batch_size=2048,
                 mini_batch_size=64,
                 hidden_width=256,
                 hidden_width2 =128,
                 lr_a=0.0002,
                 lr_c=0.0002,
                 gamma=0.99,
                 lamda=0.95,
                 epsilon=0.1,
                 K_epochs=10,
                 max_episode_steps=1000,
                 use_adv_norm=True,
                 use_state_norm=True,
                 use_reward_norm=False,
                 use_reward_scaling=True,
                 entropy_coef=0.01,
                 use_lr_decay=True,
                 use_grad_clip=True,
                 use_orthogonal_init=True,
                 set_adam_eps=True,
                 use_tanh=True,
                 chkpt_dir="/mnt/datab/home/yuanwenzheng/PICTURE1"):
        self.max_train_steps = max_train_steps
        self.evaluate_freq = evaluate_freq
        self.save_freq = save_freq
        self.policy_dist = policy_dist
        self.batch_size = batch_size
        self.mini_batch_size = mini_batch_size
        self.hidden_width = hidden_width
        self.hidden_width2 = hidden_width2
        self.lr_a = lr_a
        self.lr_c = lr_c
        self.gamma = gamma
        self.lamda = lamda
        self.epsilon = epsilon
        self.K_epochs = K_epochs
        self.use_adv_norm = use_adv_norm
        self.use_state_norm = use_state_norm
        self.use_reward_norm = use_reward_norm
        self.use_reward_scaling = use_reward_scaling
        self.entropy_coef = entropy_coef
        self.use_lr_decay = use_lr_decay
        self.use_grad_clip = use_grad_clip
        self.use_orthogonal_init = use_orthogonal_init
        self.set_adam_eps = set_adam_eps
        self.use_tanh = use_tanh
        self.max_episode_steps = max_episode_steps
        self.chkpt_dir = chkpt_dir

    def print_information(self):
        print("Maximum number of training steps:", self.max_train_steps)
        print("Evaluate the policy every 'evaluate_freq' steps:", self.evaluate_freq)
        print("Save frequency:", self.save_freq)
        print("Beta or Gaussian:", self.policy_dist)
        print("Batch size:", self.batch_size)
        print("Minibatch size:", self.mini_batch_size)
        print("The number of neurons in hidden layers of the neural network:", self.hidden_width)
        print("Learning rate of actor:", self.lr_a)
        print("Learning rate of critic:", self.lr_c)
        print("Discount factor:", self.gamma)
        print("GAE parameter:", self.lamda)
        print("PPO clip parameter:", self.epsilon)
        print("PPO parameter:", self.K_epochs)
        print("Trick 1:advantage normalization:", self.use_adv_norm)
        print("Trick 2:state normalization:", self.use_state_norm)
        print("Trick 3:reward normalization:", self.use_reward_norm)
        print("Trick 4:reward scaling:", self.use_reward_scaling)
        print("Trick 5: policy entropy:", self.entropy_coef)
        print("Trick 6:learning rate Decay:", self.use_lr_decay)
        print("Trick 7: Gradient clip:", self.use_grad_clip)
        print("Trick 8: orthogonal initialization:", self.use_orthogonal_init)
        print("Trick 9: set Adam epsilon=1e-5:", self.set_adam_eps)
        print("Trick 10: tanh activation function:", self.use_tanh)


# 下面函数用来训练网络
def train_pursuer_network(args, env, show_picture=True, pre_train=False, d_capture=0):
    Win_number = 0
    epsiode_rewards = []
    epsiode_mean_rewards = []
    # 下面将导入env环境参数
    env.d_capture = d_capture
    args.state_dim = env.observation_space.shape[0]
    args.action_dim = env.action_space.shape[0]
    args.max_action = float(env.action_space[0][1])
    # 下面将定义一个缓冲区
    replay_buffer = ReplayBuffer(args)
    # 下面将定义PPO智能体类
    pursuer_agent = PPO_continuous(args, 'pursuer')
    evader_agent = PPO_continuous(args, 'evader')
    if pre_train:
        pursuer_agent.load_checkpoint()
    evader_agent.load_checkpoint()

    # 下面开始进行训练过程
    pbar = tqdm(range(args.max_train_steps), desc="Training of pursuer", unit="episode")
    for epsiode in pbar:
        # 每个回合首先对值进行初始化
        epsiode_reward = 0.0
        done = False
        epsiode_count = 0
        # 再赋予一个新的初始状态
        s = env.reset(0)
        # 设置一个死循环，后面若跳出便在死循环中跳出
        while True:
            # 每执行一个回合，count次数加1
            epsiode_count += 1
            puruser_a, puruser_a_logprob = pursuer_agent.choose_action(s[:6])
            evader_a, evader_a_logprob = evader_agent.choose_action(s[18:24])
            # 根据参数的不同选择输出高斯分布/Beta分布调整
            if args.policy_dist == "Beta":
                puruser_action = 2 * (puruser_a - 0.5) * args.max_action
                evader_action = 2 * (evader_a - 0.5) * args.max_action
            else:
                puruser_action = puruser_a
                evader_action = evader_a

            # 下面是执行环境交互操作
            s_, r, done = env.step(puruser_action, evader_action, epsiode_count)  ## !!! 这里的环境是自己搭建的，输出每个人都不一样
            epsiode_reward += r

            # 下面考虑回合的最大运行次数(只要回合结束或者超过最大回合运行次数)
            if done or epsiode_count >= args.max_episode_steps:
                dw = True
            else:
                dw = False
            # 将经验存入replayBuffer中
            replay_buffer.store(s[:6], puruser_action, puruser_a_logprob, r, s_[:6], dw, done)
            # 重新赋值状态
            s = s_
            # 当replaybuffer尺寸到达batchsize便会开始训练
            if replay_buffer.count == args.batch_size:
                pursuer_agent.update(replay_buffer, epsiode)
                replay_buffer.count = 0
            # 如果回合结束便退出
            if done:
                epsiode_rewards.append(epsiode_reward)
                epsiode_mean_rewards.append(np.mean(epsiode_rewards))
                dis = np.linalg.norm(s_[1:3])
                if dis < d_capture:
                    Win_number += 1
                pbar.set_postfix({'获胜回合': f'{Win_number}', '奖励': f'{epsiode_reward:.1f}', '平均奖励': f'{epsiode_mean_rewards[-1]:.1f}','最终距离':f'{dis:.1f}'})
                break

    # 存储训练模型
    pursuer_agent.save_checkpoint()
    # 如果需要画图的话
    # if show_picture:
    #     fig1 = plt.figure(1)
    #     plt.plot(epsiode_rewards)
    #     plt.plot(epsiode_mean_rewards)
    #     plt.xlabel("epsiodes")
    #     plt.ylabel("pursuer_rewards")
    #     plt.title("Reward of pursuer train")
    #     plt.legend(["pursuer_epsiode_reward", "pursuer_mean_rewards"])
    #     output_file = os.path.join(save_dir, 'fig1.png')
    #     fig1.savefig(output_file)

    return pursuer_agent

def train_evader_network(args, env, show_picture=True, pre_train=False, d_capture=0):
    Win_number = 0
    epsiode_rewards = []
    epsiode_mean_rewards = []
    # 下面将导入env环境参数
    env.d_capture = d_capture
    args.state_dim = env.observation_space.shape[0]
    args.action_dim = env.action_space.shape[0]
    args.max_action = float(env.action_space[0][1])
    # 下面将定义一个缓冲区
    replay_buffer = ReplayBuffer(args)
    # 下面将定义PPO智能体类
    pursuer_agent = PPO_continuous(args, 'pursuer')
    evader_agent = PPO_continuous(args, 'evader')
    if pre_train:
        evader_agent.load_checkpoint()
    # 下面开始进行训练过程
    pbar = tqdm(range(args.max_train_steps), desc="Training of evader", unit="episode")
    for epsiode in pbar:
        # 每个回合首先对值进行初始化
        epsiode_reward = 0.0
        done = False
        epsiode_count = 0

        s = env.reset(1)

        while True:
            # 每执行一个回合，count次数加1
            epsiode_count += 1
            puruser_a, puruser_a_logprob = pursuer_agent.choose_action(s[:6])
            evader_a, evader_a_logprob = evader_agent.choose_action(s[:6])
            # 根据参数的不同选择输出是高斯分布/Beta分布调整
            if args.policy_dist == "Beta":
                puruser_action = 2 * (puruser_a - 0.5) * args.max_action
                evader_action = 2 * (evader_a - 0.5) * args.max_action
            else:
                puruser_action = puruser_a
                evader_action = evader_a
            # 下面是执行环境交互操作
            s_, r, done = env.step(puruser_action, evader_action, epsiode_count)
            epsiode_reward += r

            # 下面考虑回合的最大运行次数(只要回合结束或者超过最大回合运行次数)
            if done or epsiode_count >= args.max_episode_steps:
                dw = True
            else:
                dw = False
            # 将经验存入replayBuffer中
            replay_buffer.store(s[:6], evader_action, evader_a_logprob, r, s_[:6], dw, done)
            # 重新赋值状态
            s = s_
            # 当replaybuffer尺寸到达batchsize便会开始训练
            if replay_buffer.count == args.batch_size:
                pursuer_agent.update(replay_buffer, epsiode)
                replay_buffer.count = 0
            # 如果回合结束便退出
            if done:
                epsiode_rewards.append(epsiode_reward)
                epsiode_mean_rewards.append(np.mean(epsiode_rewards))
                dis = np.linalg.norm(s_[1:3])
                if dis > 7*d_capture:
                    Win_number += 1
                pbar.set_postfix({'获胜回合': f'{Win_number}', '奖励': f'{epsiode_reward:.1f}','平均奖励': f'{epsiode_mean_rewards[-1]:.1f}', '最终距离': f'{dis:.1f}'})
                break

    # 存储训练模型
    evader_agent.save_checkpoint()
    # 如果需要画图的话
    if show_picture:
        fig2 = plt.figure(2)
        plt.plot(epsiode_rewards)
        plt.plot(epsiode_mean_rewards)
        plt.xlabel("epsiodes")
        plt.ylabel("evader_rewards")
        plt.title("Reward of evader train")
        plt.legend(["evader_epsiode_reward", "evader_mean_rewards"])
        output_file = os.path.join("./", 'escaper_reward.png')
        fig2.savefig(output_file)
    return evader_agent

# 下面将用训练好的网络跑一次例程
def test_network(args, env, show_pictures=True, d_capture=0):
    epsiode_reward = 0.0
    done = False
    epsiode_count = 0
    pursuer_position = []
    escaper_position = []
    pursuer_velocity = []
    escaper_velocity = []
    
    total_pursuer_oil = []
    total_escaper_oil = []

    total_p_fuel_before = []
    total_p_action = [[0,0,0]]

    env.d_capture = d_capture
    args.state_dim = env.observation_space.shape[0]
    args.action_dim = env.action_space.shape[0]
    args.max_action = float(env.action_space[0][1])
    # 下面将定义PPO智能体类
    pursuer_agent = PPO_continuous(args, 'pursuer')
    evader_agent = PPO_continuous(args, 'evader')
    pursuer_agent.load_checkpoint()
    evader_agent.load_checkpoint()

    s = env.reset(3)
    pursuer_position.append(s[6:9])
    pursuer_velocity.append(s[9:12])
    escaper_position.append(s[12:15])
    escaper_velocity.append(s[15:18])
    while True:
        epsiode_count += 1
        puruser_a, puruser_a_logprob = pursuer_agent.choose_action(s[:6])
        evader_a, evader_a_logprob = evader_agent.choose_action(s[18:24])
        if args.policy_dist == "Beta":
            puruser_action = 2 * (puruser_a - 0.5) * args.max_action
            evader_action = 2 * (evader_a - 0.5) * args.max_action
        else:
            puruser_action = puruser_a
            evader_action = evader_a

        # 下面是执行环境交互操作
        s_, r, fuel_c, fuel_t, done, fuel_c_before = env.step(puruser_action, evader_action, epsiode_count,)  ## !!! 这里的环境是自己搭建的，输出每个人都不一样
        epsiode_reward += r
        pursuer_position.append(s_[6:9])
        pursuer_velocity.append(s_[9:12])
        escaper_position.append(s_[12:15])
        escaper_velocity.append(s_[15:18])
        total_pursuer_oil.append(fuel_c)
        total_escaper_oil.append(fuel_t)
        total_p_fuel_before.append(fuel_c_before)
        total_p_action.append(puruser_action)

        # 下面考虑回合的最大运行次数(只要回合结束或者超过最大回合运行次数)
        if done or epsiode_count >= args.max_episode_steps:
            dw = True
        else:
            dw = False

        s = s_
        if done :
            break
    # 下面开始画图
    if show_pictures:
        pf.plot_trajectory(pursuer_position, escaper_position)
    pursuer_position = np.array(pursuer_position)
    escaper_position = np.array(escaper_position)
    relative_distances = np.linalg.norm(pursuer_position - escaper_position, axis=1)
    # print(relative_distances[-1])
    data = {
        'p1_position_x': pursuer_position[:, 0],
        'p1_position_y': pursuer_position[:, 1],
        'p1_position_z': pursuer_position[:, 2],
        'p1_velocity_x': np.array(pursuer_velocity)[:, 0],
        'p1_velocity_y': np.array(pursuer_velocity)[:, 1],
        'p1_velocity_z': np.array(pursuer_velocity)[:, 2],

        'e_position_x': np.array(escaper_position)[:, 0],
        'e_position_y': np.array(escaper_position)[:, 1],
        'e_position_z': np.array(escaper_position)[:, 2],
        'e_velocity_x': np.array(escaper_velocity)[:, 0],
        'e_velocity_y': np.array(escaper_velocity)[:, 1],
        'e_velocity_z': np.array(escaper_velocity)[:, 2],
        'p_action':total_p_action,
        'relative_dis': np.linalg.norm(pursuer_position - escaper_position, axis=1),

        # 'p_fuel': np.array(total_p_fuel_before),
        # 'p_fuel_opt': np.array(total_pursuer_oil),
        # 'e_fuel': np.array(total_escaper_oil)
    }

    data_pd = pd.DataFrame(data)
    data_pd.to_excel("./result_1v1.xlsx", index=False)
    
    return relative_distances[0],relative_distances[-1],total_p_fuel_before[-1],total_escaper_oil[-1],len(relative_distances),total_pursuer_oil[-1]

if __name__ == "__main__":
    group_number = 0
    # 声明环境
    args = args_param(max_episode_steps=108, batch_size=108, max_train_steps=20000, K_epochs=3,
                      chkpt_dir="./one_layer")
    # 声明参数
    env = satellites(Pursuer_position=np.array([2000000, 2000000 ,1000000]),
                     Pursuer_vector=np.array([1710, 1140, 1300]),
                     Escaper_position=np.array([1850000, 2000000, 1000000]),
                     Escaper_vector=np.array([1710, 1140, 1300]),
                     d_capture=80,args=args)

    Total_start_distances = []
    Total_end_distances = []
    Total_pursuer_oil = []
    Total_escaper_oil = []
    Total_round_end = []
    Total_saved_fuel = []

    Sign = 0
    if Sign == 0:
        pursuer_agent = train_pursuer_network(args, env, show_picture=False, pre_train=True,d_capture=30)
        # evader_agent = train_evader_network(args, env, show_picture=False, pre_train=False, d_capture=5000)
    elif Sign == 1:
        test_num = 100
        progress_bar = tqdm(total=test_num)
        success_count = 0
        for i in range(test_num):
            temp1,temp2,temp3,temp4,temp5,temp6 = test_network(args, env, show_pictures=False, d_capture=50)
            progress_bar.update(1)
            if temp2 <= 80:
                success_count += 1
        print(success_count / test_num * 100, "%")
            # if temp2 <= 5000:
            #     Total_start_distances.append(temp1)
            #     Total_end_distances.append(temp2)
            #     Total_pursuer_oil.append(temp3)
            #     Total_escaper_oil.append(temp4)
            #     Total_round_end.append(temp5)
            #     Total_saved_fuel.append(temp6)
            #
            #     progress_bar.update(1)
            #     success_count += 1
            #
            # # 当成功次数达到100时退出循环
            # if success_count >= test_num:
            #     break

        # 从数据创建一个 DataFrame
        data = {
            'start_distances': Total_start_distances,
            'end_distances': Total_end_distances,
            'pursuer_oil': Total_pursuer_oil,
            'pursuer_oil_opt': Total_saved_fuel,
            'escaper_oil': Total_escaper_oil,
            'Round_end': Total_round_end
        }
        # df = pd.DataFrame(data)
        # df.to_excel('./Results/Result_evaluation.xlsx', index=False)
