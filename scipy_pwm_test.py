from scipy.integrate import solve_ivp
import numpy as np
import sympy as sp
import time as tm
import gym
from gym import spaces
from gym.utils import seeding
from os import path
import matplotlib.pyplot as plt

# def circuit_swith_off(t, y, L, C, R):
#     v = y[0]
#     iL = y[1]
#
#     dvdt = (1/C) * (iL - v/R)
#     diLdt = (-1/L) * v
#
#     return [dvdt, diLdt]
#
# def circuit_swith_on(t, y, L, C, R, E):
#     v = y[0]
#     iL = y[1]
#
#     dvdt = (1/C) * (iL - v/R)
#     diLdt = (1/L) * (E - v)
#
#     return [dvdt, diLdt]
#
#
# t_ini = 0
# y_ini = [0, 0]
# t_end = 0.004
# t_span = [t_ini, t_end]
#
# t_list = np.linspace(t_ini, t_end, 201)
#
# L_value = 0.35e-3; C_value = 0.22e-3; R_value = 10; E_value = 100
# # ansivp = solve_ivp(circuit_swith_off, t_span, y_ini, t_eval=t_list, args=(L, C, R), rtol=1e-12, atol=1e-14)
# start_time1 = tm.time()
#
# ansivp = solve_ivp(circuit_swith_on, t_span, y_ini, t_eval=t_list, args=(L_value, C_value, R_value, E_value),
#                    rtol=1e-6, atol=1e-8)
# #default rtol -12 atol -14
# elapsed_time1 = tm.time() - start_time1
#
# sig = 0.5
# # PWM波形のスイッチングタイミングを計算
# time_turn_on = ((1 - sig) / (2 * 1)) * (20e-6 * 0.5)
# time_turn_off = ((sig + 1) / (2 * 1)) * (20e-6 * 0.5) + 20e-6 * 0.5

class BuckConverterEnvWithScipy(gym.Env):

    def __init__(self, dt=20e-6, E_ini=10, R_ini=10, L_ini=0.001, C_ini=1e-6, career_amp=1, smooth_scale=1):

        self.state_list = [np.array([0.0, 0.0, 0.0], dtype=np.float32)]  # iL, V_out, time の順番で記録
        self.state_list_smooth = [np.array([0.0, 0.0, 0.0], dtype=np.float32)]  # iL, V_out, time の順番で記録
        self.observation_list = [np.array([0.0, 0.0], dtype=np.float32),    # iL, V_out
                                 np.array([0.0, 0.0], dtype=np.float32),
                                 np.array([0.0, 0.0], dtype=np.float32)]
        self.time_index = 0
        self.E = E_ini
        self.R_ini = R_ini
        self.R_now = self.R_ini
        self.L = L_ini
        self.C = C_ini
        self.CAREER_AMP = career_amp
        self.SMOOTH_SCALE = smooth_scale
        self.dt_cont = dt
        self.v_out_command = 25 / self.E
        self.done = False
        self.smooth_flag = True if smooth_scale > 1 else False
        self.normalize_ampere = 10
        self.steady_state_flag = int(False)

        # 行動空間の定義
        # high = np.array([1], dtype=np.float32)
        self.action_space = spaces.Box(low=-self.CAREER_AMP,
                                       high=self.CAREER_AMP,
                                       shape=(1,),
                                       dtype=np.float32)

        # 状態空間の定義
        high = np.array([np.finfo(np.float32).max,
                         np.finfo(np.float32).max,
                         np.finfo(np.float32).max,
                         np.finfo(np.float32).max,
                         np.finfo(np.float32).max,
                         np.finfo(np.float32).max,
                         np.finfo(np.float32).max,
                         1],
                        dtype=np.float32)
        self.observation_space = spaces.Box(low=-high, high=high, dtype=np.float32)

        self.seed()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, sig):

        if np.isnan(sig):
            raise Exception("invalid value encountered in input value")

        sig = np.clip(sig, a_min=-self.CAREER_AMP, a_max=self.CAREER_AMP)[0]
        iL_now, v_out_now, time_pre = self.state_list[self.time_index]

        # PWM波形のスイッチングタイミングを計算
        time_turn_on = ((self.CAREER_AMP - sig) / (2 * self.CAREER_AMP)) * (self.dt_cont * 0.5)
        time_turn_off = ((sig + self.CAREER_AMP) / (2 * self.CAREER_AMP)) * (self.dt_cont * 0.5) + self.dt_cont * 0.5

        def circuit_function_swith_off(t, y, L, C, R):
            v = y[0]
            iL = y[1]

            dvdt = (1 / C) * (iL - v / R)
            diLdt = (-1 / L) * v

            return [dvdt, diLdt]

        def circuit_function_swith_on(t, y, L, C, R, E):
            v = y[0]
            iL = y[1]

            dvdt = (1 / C) * (iL - v / R)
            diLdt = (1 / L) * (E - v)

            return [dvdt, diLdt]

        if not time_turn_on == 0:
            t_ini = 0
            y_ini = np.array([v_out_now, iL_now]).reshape(-1)
            t_end = time_turn_on
            t_span = [t_ini, t_end]
            t_list = [t_end]
            ansivp = solve_ivp(circuit_function_swith_off, t_span, y_ini, t_eval=t_list,
                               args=(self.L, self.C, self.R_ini), rtol=1e-6, atol=1e-8)
            v_out_now = (ansivp.y)[0]
            iL_now = (ansivp.y)[1]

        if not time_turn_on == time_turn_off:
            t_ini = time_turn_on
            y_ini = np.array([v_out_now, iL_now]).reshape(-1)
            t_end = time_turn_off
            t_span = [t_ini, t_end]
            t_list = [t_end]
            ansivp = solve_ivp(circuit_function_swith_on, t_span, y_ini, t_eval=t_list,
                               args=(self.L, self.C, self.R_ini, self.E), rtol=1e-6, atol=1e-8)
            v_out_now = (ansivp.y)[0]
            iL_now = (ansivp.y)[1]

        if not time_turn_off == self.dt_cont:
            t_ini = time_turn_off
            y_ini = np.array([v_out_now, iL_now]).reshape(-1)
            t_end = self.dt_cont
            t_span = [t_ini, t_end]
            t_list = [t_end]
            ansivp = solve_ivp(circuit_function_swith_off, t_span, y_ini, t_eval=t_list,
                               args=(self.L, self.C, self.R_ini), rtol=1e-6, atol=1e-8)
            v_out_now = (ansivp.y)[0]
            iL_now = (ansivp.y)[1]

        self.time_index += 1

        state = np.array([iL_now, v_out_now, self.time_index * self.dt_cont], dtype=np.float32)

        # ガウス分布に従うノイズをフィードバックにのみ付与
        # iL_now += np.random.normal(loc=0.0, scale=1e-3, size=None)
        # v_out_now += np.random.normal(loc=0.0, scale=1e-2, size=None)

        # 状態値としての負荷電圧は電源電圧で規格化、インダクタ電流は10A上限として規格化(仮)
        obs = np.array([iL_now / self.normalize_ampere, v_out_now / self.E], dtype=np.float32).reshape(-1)
        self.state_list.append(state)
        self.observation_list.append(obs)

        return self.__get_observation_for_test(), self.__get_reward(), False, {}

    def reset(self, il_ini=0.0, v_out_ini=0.0, v_out_command=25):

        state_ini = np.array([il_ini, v_out_ini, 0.0], dtype=np.float32)
        self.state_list = [state_ini]
        self.state_list_smooth = [state_ini]
        self.time_index = 0
        self.v_out_command = v_out_command / self.E
        self.steady_state_flag = int(False)

        il_ini /= self.normalize_ampere
        v_out_ini /= self.E
        self.observation_list = [np.array([il_ini, v_out_ini], dtype=np.float32),
                                 np.array([il_ini, v_out_ini], dtype=np.float32),
                                 np.array([il_ini, v_out_ini], dtype=np.float32)]
        self.R_now = self.R_ini

        return self.__get_observation_for_test()

    def render(self, mode='human'):

        iL_list = [temp[0] for temp in self.state_list]
        v_out_list = [temp[1] for temp in self.state_list]
        time_list = [temp[2] * 1000 for temp in self.state_list]

        fig = plt.figure()
        ax1 = fig.add_subplot()
        ax2 = ax1.twinx()

        ax1.plot(time_list, v_out_list, color="orange", label="v out")
        ax2.plot(time_list, iL_list, color="blue", label="iL")

        ax1.set_xlabel("t [ms]")
        ax1.set_ylabel("v out [V]")
        ax2.set_ylabel("iL [A]")

        h1, l1 = ax1.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        ax1.legend(h1+h2, l1+l2, loc='lower right')

        return fig, ax1, ax2

    def render_smooth(self):

        iL_list = [temp[0] for temp in self.state_list_smooth]
        v_out_list = [temp[1] for temp in self.state_list_smooth]
        time_list = [temp[2] * 1000 for temp in self.state_list_smooth]

        fig = plt.figure()
        ax1 = fig.add_subplot()
        ax2 = ax1.twinx()

        ax1.plot(time_list, v_out_list, color="orange", label="v out")
        ax2.plot(time_list, iL_list, color="blue", label="iL")

        ax1.set_xlabel("t [ms]")
        ax1.set_ylabel("v out [V]")
        ax2.set_ylabel("iL [A]")

        h1, l1 = ax1.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        ax1.legend(h1+h2, l1+l2, loc='lower right')

        return fig, ax1, ax2

    def get_status_list(self):
        return tuple(self.state_list)

    def get_smooth_status_list(self):
        return tuple(self.state_list_smooth)

    def __get_observation_for_test(self):
        observation = self.observation_list[self.time_index + 2]
        return observation

    def __get_observation(self):
        observation = np.ravel(np.zeros((1, 7), dtype=np.float32))
        observation[0: 6] = np.concatenate([self.observation_list[self.time_index],
                                            self.observation_list[self.time_index + 1],
                                            self.observation_list[self.time_index + 2]], 0)
        observation[6] = self.v_out_command
        return observation

    def __get_observation2(self):
        observation = np.ravel(np.zeros((1, 5), dtype=np.float32))
        # 3ステップ分の負荷電流のリスト
        iR_liston3step = [self.observation_list[self.time_index][1],
                          self.observation_list[self.time_index + 1][1],
                          self.observation_list[self.time_index + 2][1]]
        observation[0: 3] = iR_liston3step
        observation[3] = self.observation_list[self.time_index + 2][0]  # 現在のインダクタ電流を代入
        observation[4] = self.v_out_command    # 電流の指令値を代入
        return observation

    def __get_reward(self):

        reward = 0
        error_limit_ratio = 0.02
        upper_limit = self.v_out_command * (1 + error_limit_ratio)
        under_limit = self.v_out_command * (1 - error_limit_ratio)

        return reward

    def change_load(self, load_value):
        assert 0 < load_value
        self.R_now = load_value