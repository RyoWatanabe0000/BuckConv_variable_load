import sys, os
import numpy as np
import torch.optim as optim

from pfrl import explorers
from pfrl import q_functions, replay_buffers, utils
from pfrl.agents import DQN, DoubleDQN
from pfrl.q_functions import FCQuadraticStateQFunction

import datetime
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt
from buck_converter_env_gradual_load_change_with_scipy import BuckConverterEnv

# 図用の初期設定
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

N_LOOP = 100
N_EPISODE = 3000
MAX_EPI_LEN = int(TIME_LIMIT / TIME_contorl)
GAMMA = 0.99
N_HIDDON_CHANNEL = 100
N_HIDDON_LAYER = 2
BUFFER_SIZE = 10**5
REPLAY_START_SIZE = 5000
TARGET_INTERVAL = 1000
UPDATE_INTERVAL = 1
MINIBATCH_SIZE = 32
LEARNING_RATE = 6e-4
SAVE_DIR = f'agents64_6e-4_gauss036_with_steadyflag_new'
START_FLAG = True
break_flag = False
RESISTANCE_CHANGE_RATE = 25000

env = BuckConverterEnv(dt=TIME_contorl, E_ini=E, R_ini=R, L_ini=L, C_ini=C, career_amp=1, smooth_scale=1)
env_for_test = BuckConverterEnv(dt=TIME_contorl, E_ini=E, R_ini=R, L_ini=L, C_ini=C, career_amp=1, smooth_scale=50)

obs_space = env.observation_space
obs_size = obs_space.low.size
action_space = env.action_space
action_size = action_space.low.size

def write_parameter_to_CSV(SAVE_PATH):
    para_name_list = ["N_EPISODE", "MAX_EPI_LEN", "GAMMA", "N_HIDDON_CHANNEL", "N_HIDDON_LAYER",
                      "BUFFER_SIZE", "REPLAY_START_SIZE", "TARGET_INTERVAL", "UPDATE_INTERVAL",
                      "MINIBATCH_SIZE","LEARNING_RATE", "dt", "E", "L", "R", "C"]
    para_list = [N_EPISODE, MAX_EPI_LEN, GAMMA, N_HIDDON_CHANNEL, N_HIDDON_LAYER, BUFFER_SIZE,
                 REPLAY_START_SIZE, TARGET_INTERVAL, UPDATE_INTERVAL, MINIBATCH_SIZE, LEARNING_RATE,
                 TIME_contorl, E, L, R, C]

    para_series1 = pd.Series(data=para_name_list)
    para_series2 = pd.Series(data=para_list)
    para_df = pd.concat([para_series1, para_series2], axis=1)

    para_df.to_csv(f'{SAVE_PATH}/parameter.csv', mode='a', header=False, index=False)

def save_reward_graph_and_reward_data(R_list:list, SAVE_PATH):

    moving_ave = lambda x, w: np.convolve(x, np.ones(w), 'same') / w

    episode_array = range(1, N_EPISODE + 1) if not break_flag else range(1, 3001)

    plt.figure()
    plt.plot(episode_array, R_list, color="blue", alpha=0.2)
    plt.plot(episode_array, moving_ave(R_list, 10), color="blue", alpha=1)
    plt.title("Reward")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.savefig(f"{SAVE_PATH}/reward_graph.png", bbox_inches="tight", pad_inches=0.05, dpi=600)
    plt.close()

    reward_series = pd.Series(data=R_list)

    reward_series.to_csv(f'{SAVE_PATH}/reward.csv', mode='a', header=False, index=False)

def save_status_graph_and_data_to_csv(action_list:list, status_list:list, status_list_smooth:list, SAVE_PATH):

    iL_list = [temp[0] for temp in status_list]
    v_out_list = [temp[1] for temp in status_list]
    time_list = [temp[2] * 1000 for temp in status_list]

    iL_list_smooth = [temp[0] for temp in status_list_smooth]
    v_out_list_smooth = [temp[1] for temp in status_list_smooth]
    time_list_smooth = [temp[2] * 1000 for temp in status_list_smooth]

    df = pd.DataFrame(list(zip(time_list_smooth, iL_list_smooth, v_out_list_smooth)),
                      columns=["time[s]", "iL[A]", "v out[V]"], index=None)
    df.to_csv(f"{SAVE_PATH}/current_data.csv", index=False)

    fig = plt.figure()
    ax1 = fig.add_subplot()
    ax2 = ax1.twinx()
    ax1.plot(time_list, v_out_list, color="red", label="$V_{out}$")
    ax2.plot(time_list, iL_list, color="blue", label="$i_{L}$")
    ax1.set_xlabel("t [ms]")
    ax1.set_ylabel("$V_{out}$ [V]")
    ax2.set_ylabel("$i_{L}$ [A]")
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc='lower right')
    fig.savefig(f"{SAVE_PATH}/current_graph.png", bbox_inches="tight", pad_inches=0.05, dpi=600)
    plt.close()

    fig = plt.figure()
    ax1 = fig.add_subplot()
    ax2 = ax1.twinx()
    ax1.plot(time_list_smooth, v_out_list_smooth, color="red", label="$V_{out}$")
    ax2.plot(time_list_smooth, iL_list_smooth, color="blue", label="$i_{L}$")
    ax1.set_xlabel("t [ms]")
    ax1.set_ylabel("$V_{out}$ [V]")
    ax2.set_ylabel("$i_{L}$ [A]")
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc='lower right')
    fig.savefig(f"{SAVE_PATH}/current_graph.png", bbox_inches="tight", pad_inches=0.05, dpi=600)
    plt.close()

    del time_list[-1]   #要素数をあわせるために削除
    plt.figure()
    plt.plot(time_list, action_list)
    plt.xlabel("time [ms]")
    plt.ylabel("signal")
    plt.savefig(f"{SAVE_PATH}/action_graph.png", bbox_inches="tight", pad_inches=0.05, dpi=600)
    plt.close()

def save_evaluation_graph(eval_reward_list:list, SAVE_PATH):

    episode_list = [temp[0] for temp in eval_reward_list]
    eval_list = [temp[1] for temp in eval_reward_list]

    plt.figure()
    plt.plot(episode_list, eval_list)
    plt.xlabel("episode")
    plt.ylabel("evaluation value")
    plt.ylim(0, 700)
    plt.savefig(f"{SAVE_PATH}/evaluation.png", bbox_inches="tight", pad_inches=0.05, dpi=600)
    plt.close()

def clip_action_filter(a):
    return np.clip(a, action_space.low, action_space.high)

utils.env_modifiers.make_action_filtered(env, clip_action_filter)
utils.env_modifiers.make_action_filtered(env_for_test, clip_action_filter)


def main():
    global START_FLAG, break_flag
    eval_reward_list = []
    best_eval_value = 0
    break_flag = False

    # フォルダを作成
    now = datetime.datetime.now()
    if START_FLAG:
        SAVE_PATH = f"{SAVE_DIR}/{now:%m%d%H%M}_INI"
    else:
        SAVE_PATH = f"{SAVE_DIR}/{now:%m%d%H%M}"
    if not os.path.exists(SAVE_PATH):
        os.makedirs(SAVE_PATH)
    if not os.path.exists(f"{SAVE_PATH}/best_model"):
        os.makedirs(f"{SAVE_PATH}/best_model")
    if not os.path.exists(f"{SAVE_PATH}/last_model"):
        os.makedirs(f"{SAVE_PATH}/last_model")

    q_func = FCQuadraticStateQFunction(
        obs_size,
        action_size,
        n_hidden_channels=N_HIDDON_CHANNEL,
        n_hidden_layers=N_HIDDON_LAYER,
        action_space=action_space,
    )

    explorer = explorers.AdditiveGaussian(scale=0.36, low=env.action_space.low, high=env.action_space.high)

    opt = optim.Adam(q_func.parameters(), lr=LEARNING_RATE, eps=1e-8)

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
        update_interval=UPDATE_INTERVAL,
        minibatch_size=MINIBATCH_SIZE,
        phi=phi
    )

    # agent.load("agents64_3e-4_gauss/10110817/best_model")

    def eval_model():
        total_reward = 0
        with agent.eval_mode():
            for i in range(1):
                vcc = 25 + i * 10
                obs = env_for_test.reset(v_out_command=vcc, resistance_change_rate=RESISTANCE_CHANGE_RATE)
                R = 0
                t = 0
                while True:
                    if t == int(MAX_EPI_LEN / 2):
                        env_for_test.set_load_value(5)
                    action = agent.act(obs)
                    obs, r, done, _ = env_for_test.step(action)
                    R += r
                    t += 1
                    reset = t == MAX_EPI_LEN
                    # agent.observe(obs, r, done, reset)
                    if done or reset:
                        break
                total_reward += R
        return total_reward

    # 学習開始
    print("start", end='')
    R_list = []
    for episode in range(1, N_EPISODE + 1):

        v_out_ref = np.random.uniform(low=20, high=60)
        v_out_ref = 25
        load_change_index = int(np.random.uniform(low=100, high=125))
        # while True:
        #     second_load_value = np.random.uniform(low=5.0, high=15.0)
        #     # 負荷が重くなった場合に定常状態で必要な電流が8Aを超えないように負荷の値を選択する
        #     if v_out_ref / 8 < second_load_value:
        #         break
        second_load_value = 5

        # obs = env.reset(il_ini=v_out_ref/10, v_out_ini=v_out_ref, v_out_command=v_out_ref)
        obs = env.reset(v_out_command=v_out_ref, resistance_change_rate=RESISTANCE_CHANGE_RATE)
        R = 0
        step = 0

        while True:
            if step == load_change_index:
                env.set_load_value(second_load_value)

            action = agent.act(obs)

            try:
                obs, reward, done, _ = env.step(action)
            except Exception as e:
                print(e)
                return

            R += reward
            step += 1
            reset = step == MAX_EPI_LEN
            agent.observe(obs, reward, done, reset)
            if done or reset:
                break

        R_list.append(R)

        if episode % 10 == 0:
            eval_reward_list.append([episode, eval_model()])
            # try:
            #     save_evaluation_graph(eval_reward_list, SAVE_PATH)
            # except Exception as e:
            #     pass

            if best_eval_value < eval_reward_list[-1][1] or episode == 10:
                best_eval_value = eval_reward_list[-1][1]
                agent.save(f"{SAVE_PATH}/best_model")

        if episode % 500 == 0:
            print(f"{episode}:{int(eval_reward_list[-1][1])}→→", end='')

        # if episode == 3000:     # エピソード2000時点ですべての報酬が負ならば学習を止める
        #     eval_reward_list_without_epi = [temp[1] for temp in eval_reward_list]
        #     eval_reward_numpy = np.array(eval_reward_list_without_epi, dtype=np.float32)
        #     if np.all(eval_reward_numpy < 0):
        #         break_flag = True
        #         break

    # 最終モデルを保存
    agent.save(f"{SAVE_PATH}/last_model")
    print("Finish", end=" ")

    #学習結果のテスト
    # 最終モデルをロード
    agent.load(f"{SAVE_PATH}/best_model")

    with agent.eval_mode():
        obs = env_for_test.reset(v_out_command=25, resistance_change_rate=RESISTANCE_CHANGE_RATE)
        R = 0
        t = 0
        action_list = []
        while True:
            if t == int(MAX_EPI_LEN / 2):
                env_for_test.set_load_value(5)

            action = agent.act(obs)
            obs, r, done, _ = env_for_test.step(action)
            R += r
            t += 1
            reset = t == MAX_EPI_LEN
            # agent.observe(obs, r, done, reset)
            action_list.append(action)
            if done or reset:
                break

    status_list = env_for_test.get_status_list()
    status_list_smooth = env_for_test.get_smooth_status_list()

    # 各種データの保存を行う
    write_parameter_to_CSV(SAVE_PATH)
    save_status_graph_and_data_to_csv(action_list, status_list, status_list_smooth, SAVE_PATH)
    save_reward_graph_and_reward_data(R_list, SAVE_PATH)
    save_evaluation_graph(eval_reward_list, SAVE_PATH)
    env_for_test.save_v_error_graph(SAVE_PATH)

    massage = "正常終了" if not break_flag else " 中断"
    print(f"評価報酬 {R:> 5.1f} 保存ファイル {SAVE_PATH} {massage}")
    START_FLAG = False
    del q_func, explorer, opt, rbuf, agent

for _ in range(N_LOOP):
    main()