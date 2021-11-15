# coding=utf-8
# Created by watanabe at 2021/10/08
import gym
import numpy as np
import torch
import torch.optim as optim

import pfrl
from pfrl import experiments, explorers
from pfrl import nn as pnn
from pfrl import q_functions, replay_buffers, utils
from pfrl.agents.dqn import DQN

# from torch.utils.tensorboard import SummaryWriter
import datetime
from tqdm import tqdm
# from buck_converter_env import BuckConverterEnv
from buck_converter_env_gradual_load_change_with_scipy import BuckConverterEnv
import matplotlib.pyplot as plt

# 図用の初期設定
plt.rcParams['axes.grid'] = False
plt.rcParams['font.family'] ='sans-serif'   #使用するフォント
plt.rcParams['xtick.direction'] = 'in'      #x軸の目盛線が内向き('in')か外向き('out')か双方向か('inout')
plt.rcParams['ytick.direction'] = 'in'      #y軸の目盛線が内向き('in')か外向き('out')か双方向か('inout')
plt.rcParams['xtick.major.width'] = 1.0     #x軸主目盛り線の線幅
plt.rcParams['ytick.major.width'] = 1.0     #y軸主目盛り線の線幅
plt.rcParams['font.size'] = 12               #フォントの大きさ
plt.rcParams['axes.linewidth'] = 1.0        # 軸の線幅edge linewidth。囲みの太さ
plt.rcParams['axes.xmargin'] = 0
plt.rcParams['axes.ymargin'] = 0

TIME_LIMIT = 4e-3   #シミュレーション終了時間
FREQ_control = 50e+3    #制御周波数
TIME_contorl = 1/FREQ_control   #制御周期

# 素子値
E = 100
L = 0.50e-3
R = 25
C = 330e-6

N_EPISODE = 2000
MAX_EPI_LEN = int(TIME_LIMIT / TIME_contorl)
GAMMA = 0.99
N_HIDDON_CHANNEL = 100
N_HIDDON_LAYER = 2
BUFFER_SIZE = 10**5
REPLAY_START_SIZE = 5000
TARGET_INTERVAL = 1000
UPDATE_INTERVAL = 1
MINIBATCH_SIZE = 32
RESISTANCE_CHANGE_RATE = 50000
AGENT_DIR = f'agents64_6e-4_gauss036_with_steadyflag_new/11150551/best_model'

env = BuckConverterEnv(dt=TIME_contorl, E_ini=E, R_ini=R, L_ini=L, C_ini=C, career_amp=1, smooth_scale=50)

obs_space = env.observation_space
obs_size = obs_space.low.size
action_space = env.action_space
action_size = action_space.low.size

# def clip_action_filter(a):
#     return np.clip(a, action_space.low, action_space.high)
#
# utils.env_modifiers.make_action_filtered(env, clip_action_filter)

q_func = q_functions.FCQuadraticStateQFunction(
    obs_size,
    action_size,
    n_hidden_channels=N_HIDDON_CHANNEL,
    n_hidden_layers=N_HIDDON_LAYER,
    action_space=action_space,
)

explorer = explorers.AdditiveGaussian(scale=0.36, low=env.action_space.low, high=env.action_space.high)

opt = optim.Adam(q_func.parameters())

rbuf = replay_buffers.ReplayBuffer(BUFFER_SIZE)

phi = lambda x: x.astype(np.float32, copy=False)

agent = DQN(
    q_func,
    opt,
    rbuf,
    gpu=-1,
    gamma=GAMMA,
    explorer=explorer,
    replay_start_size=REPLAY_START_SIZE,
    target_update_interval=TARGET_INTERVAL,
    update_interval=1,
    minibatch_size=MINIBATCH_SIZE,
    phi=phi
)

agent.load(AGENT_DIR)

with agent.eval_mode():

    obs = env.reset(v_out_command=25,
                    resistance_change_rate=RESISTANCE_CHANGE_RATE)
    R = 0
    t = 0
    action_list = []
    while True:
        if t == 100:
            env.set_load_value(5)
        action = agent.act(obs)
        obs, r, done, _ = env.step(action)
        R += r
        t += 1
        reset = t == MAX_EPI_LEN
        # agent.observe(obs, r, done, reset)
        action_list.append(action)
        if done or reset:
            break

    # plt.figure()
    # plt.plot(range(1, MAX_EPI_LEN + 1), action_list)
    # env.render()
    fig, ax1, ax2 = env.render_smooth()
    # ax2.set_ylim(0, 12)

    # plt.xlim(1.2, 1.25)
    # plt.ylim(1.5, 3.5)
    plt.show()