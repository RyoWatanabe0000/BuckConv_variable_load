import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
from os import path
import sympy as sp
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp


class BuckConverterEnv(gym.Env):

    def __init__(self, dt=20e-6, E_ini=10, R_ini=10, L_ini=0.001, C_ini=1e-6, career_amp=1, smooth_scale=1,
                 resistance_change_rate=25000, clip_ampere_value=15):

        self.state_list = [np.array([0.0, 0.0, 0.0], dtype=np.float32)]  # iL, V_out, time の順番で記録
        self.state_list_smooth = [np.array([0.0, 0.0, 0.0], dtype=np.float32)]  # iL, V_out, time の順番で記録
        self.observation_list = [np.array([0.0, 0.0], dtype=np.float32),  # iL, V_out
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
        self.normalize_ampere = clip_ampere_value
        self.during_load_change_flag = False
        self.resistance_change_rate = resistance_change_rate  # ohm/s
        self.load_transition_time = 0   # 負荷変動にかかる時間 [s]
        self.load_transition_num_step = 0   # 負荷変動に必要なステップ数
        self.load_change_steps_remaining = 0    # 負荷変動に必要な残りステップ数
        self.load_transition_num_elapsed_step = 0   # 負荷変動を開始してから経過しているステップ数
        self.load_value_list = [self.R_ini]
        self.rtol = 1e-6   # scipy用設定　相対許容誤差
        self.atol = 1e-8   # scipy用設定　絶対許容誤差

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
                         np.finfo(np.float32).max],
                        dtype=np.float32)
        self.observation_space = spaces.Box(low=-high, high=high, dtype=np.float32)

        self.seed()

        """
        関数の算出
        """
        # シンボルの定義
        t = sp.symbols('t', real=True)
        iL_ini, iR_ini, Q_ini, v_out_ini = sp.symbols('iL_ini iR_ini Q_ini v_out_ini', real=True)
        C1, C2 = sp.symbols('C1 C2')
        R, E, C, L = sp.symbols('R E C L')
        Q = sp.Function('Q')(t)

        R_ini = sp.symbols('R_ini')
        m_r = sp.symbols('m_R')

        # SW off
        # コンデンサに蓄えられる電荷Qに関する二階線形微分方程式を解く
        ode_Q_off = sp.Eq(sp.diff(Q, t, 2) + (1 / (C * R)) * sp.diff(Q, t, 1) + (1 / (L * C)) * Q, 0)
        self.Q_func_off = sp.dsolve(ode_Q_off, hint='nth_linear_constant_coeff_homogeneous')
        self.v_out_func_off = self.Q_func_off.rhs / C
        self.iL_func_off = (1 / (C * R)) * self.Q_func_off.rhs + sp.diff(self.Q_func_off.rhs, t, 1)

        # 定数C1C2についての方程式を立式
        self.C_func_off = sp.solve([sp.Eq(self.v_out_func_off, v_out_ini),
                                    sp.Eq(self.iL_func_off, iL_ini)],
                                   [C1, C2], hint="best")

        # SW on
        # コンデンサに蓄えられる電荷Qに関する二階線形微分方程式を解く
        ode_Q_on = sp.Eq(sp.diff(Q, t, 2) + (1 / (C * R)) * sp.diff(Q, t, 1) + (1 / (L * C)) * Q, E / L)
        self.Q_func_on = sp.dsolve(ode_Q_on, hint='nth_linear_constant_coeff_undetermined_coefficients')
        self.v_out_func_on = self.Q_func_on.rhs / C
        self.iL_func_on = self.Q_func_on.rhs / (C * R) + sp.diff(self.Q_func_off.rhs, t, 1)

        # 定数C1C2についての方程式を立式
        self.C_func_on = sp.solve([sp.Eq(self.v_out_func_on, v_out_ini),
                                   sp.Eq(self.iL_func_on, iL_ini)],
                                  [C1, C2], hint="best")

        # print(sp.latex(self.v_out_func_off))
        # print(sp.latex(self.iL_func_off))
        # print(sp.latex(self.C_func_off))
        # print("")
        # print(sp.latex(self.v_out_func_on))
        # print(sp.latex(self.iL_func_on))
        # print(sp.latex(self.C_func_on))
        # print("")

        self.v_out_func_off = self.v_out_func_off.subs([(L, self.L), (C, self.C), (E, self.E)])
        self.v_out_func_on = self.v_out_func_on.subs([(L, self.L), (C, self.C), (E, self.E)])
        self.iL_func_off = self.iL_func_off.subs([(L, self.L), (C, self.C), (E, self.E)])
        self.iL_func_on = self.iL_func_on.subs([(L, self.L), (C, self.C), (E, self.E)])
        self.C_func_off[C1] = self.C_func_off[C1].subs([(L, self.L), (C, self.C), (E, self.E)])
        self.C_func_off[C2] = self.C_func_off[C2].subs([(L, self.L), (C, self.C), (E, self.E)])
        self.C_func_on[C1] = self.C_func_on[C1].subs([(L, self.L), (C, self.C), (E, self.E)])
        self.C_func_on[C2] = self.C_func_on[C2].subs([(L, self.L), (C, self.C), (E, self.E)])

        # 電流と任意定数Cの関数を扱いやすい形に変換(numpyのufuncに変換)
        self.v_out_ufunc_off = sp.lambdify((t, C1, C2, R), self.v_out_func_off, "numpy")
        self.iL_ufunc_off = sp.lambdify((t, C1, C2, R), self.iL_func_off, "numpy")
        self.v_out_ufunc_on = sp.lambdify((t, C1, C2, R), self.v_out_func_on, "numpy")
        self.iL_ufunc_on = sp.lambdify((t, C1, C2, R), self.iL_func_on, "numpy")

        self.C1_ufunc_off = sp.lambdify((t, iL_ini, v_out_ini, R), self.C_func_off[C1], "numpy")
        self.C1_ufunc_on = sp.lambdify((t, iL_ini, v_out_ini, R), self.C_func_on[C1], "numpy")
        self.C2_ufunc_off = sp.lambdify((t, iL_ini, v_out_ini, R), self.C_func_off[C2], "numpy")
        self.C2_ufunc_on = sp.lambdify((t, iL_ini, v_out_ini, R), self.C_func_on[C2], "numpy")

        # print(sp.latex(self.v_out_func_off))
        # print(sp.latex(self.iL_func_off))
        # print(sp.latex(self.C_func_off))
        # print("")
        # print(sp.latex(self.v_out_func_on))
        # print(sp.latex(self.iL_func_on))
        # print(sp.latex(self.C_func_on))
        # print("")

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, sig):

        # sigに対するエラー処理とクリップ
        if np.isnan(sig):
            raise Exception("invalid value encountered in input value")
        sig = np.clip(sig, a_min=-self.CAREER_AMP, a_max=self.CAREER_AMP)[0]

        # 負荷変動していない時間はsympy、している時間はscipyで計算を行う
        if not self.during_load_change_flag:
            iL_now, v_out_now = self.__cal_circuit_with_sympy(sig)
        else:
            iL_now, v_out_now = self.__cal_circuit_with_scipy(sig)

        self.time_index += 1

        state = np.array([iL_now, v_out_now, self.time_index * self.dt_cont], dtype=np.float32)

        # ガウス分布に従うノイズをフィードバックにのみ付与
        iL_now += np.random.normal(loc=0.0, scale=3e-3, size=None)
        v_out_now += np.random.normal(loc=0.0, scale=2.5e-2, size=None)

        # 状態値としての負荷電圧は電源電圧で規格化、インダクタ電流は10A上限として規格化(仮)
        obs = np.array([iL_now / self.normalize_ampere, v_out_now / self.E], dtype=np.float32).reshape(-1)
        self.state_list.append(state)
        self.observation_list.append(obs)
        self.load_value_list.append(self.R_now)     # 現在の負荷の値を記録

        return self.__get_observation_for_test(), self.__get_reward(), False, {}

    def reset(self, il_ini=0.0, v_out_ini=0.0, v_out_command=25, resistance_change_rate=25000):

        state_ini = np.array([il_ini, v_out_ini, 0.0], dtype=np.float32)
        self.state_list = [state_ini]
        self.state_list_smooth = [state_ini]
        self.time_index = 0
        self.v_out_command = v_out_command / self.E

        il_ini /= self.normalize_ampere
        v_out_ini /= self.E
        self.observation_list = [np.array([il_ini, v_out_ini], dtype=np.float32),
                                 np.array([il_ini, v_out_ini], dtype=np.float32),
                                 np.array([il_ini, v_out_ini], dtype=np.float32)]
        self.R_now = self.R_ini

        # 負荷変動用定数のリセット
        self.during_load_change_flag = False
        self.resistance_change_rate = resistance_change_rate  # ohm/s
        self.load_transition_time = 0
        self.load_transition_num_step = 0
        self.load_change_steps_remaining = 0

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
        ax1.legend(h1 + h2, l1 + l2, loc='lower right')

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
        ax1.legend(h1 + h2, l1 + l2, loc='lower right')

        return fig, ax1, ax2

    def get_status_list(self):
        return tuple(self.state_list)

    def get_smooth_status_list(self):
        if self.smooth_flag:
            return tuple(self.state_list_smooth)
        else:
            return False

    def __get_observation_for_test(self):
        observation = self.observation_list[self.time_index + 2]
        return observation

    # 3ステップの負荷電圧とインダクタ電流を状態値に含む
    def __get_observation(self):
        observation = np.ravel(np.zeros((1, 7), dtype=np.float32))
        observation[0: 6] = np.concatenate([self.observation_list[self.time_index],
                                            self.observation_list[self.time_index + 1],
                                            self.observation_list[self.time_index + 2]], 0)
        observation[6] = self.v_out_command
        return observation

    # 3ステップの負荷電圧と現在のインダクタ電流を状態値に含む
    def __get_observation2(self):
        observation = np.ravel(np.zeros((1, 5), dtype=np.float32))
        # 3ステップ分の負荷電流のリスト
        iR_liston3step = [self.observation_list[self.time_index][1],
                          self.observation_list[self.time_index + 1][1],
                          self.observation_list[self.time_index + 2][1]]
        observation[0: 3] = iR_liston3step
        observation[3] = self.observation_list[self.time_index + 2][0]  # 現在のインダクタ電流を代入
        observation[4] = self.v_out_command  # 電流の指令値を代入
        return observation

    def __get_reward(self):

        reward = 0
        error_limit_ratio = 0.01
        upper_limit = self.v_out_command * (1 + error_limit_ratio)
        under_limit = self.v_out_command * (1 - error_limit_ratio)

        prepre_iL, prepre_v_out = self.observation_list[self.time_index]
        pre_iL, pre_v_out = self.observation_list[self.time_index + 1]
        current_iL, current_v_out = self.observation_list[self.time_index + 2]

        if pre_v_out <= under_limit and current_v_out <= under_limit:
            if current_v_out > pre_v_out:
                reward += 0.01

        if under_limit < prepre_v_out < upper_limit and \
                under_limit < pre_v_out < upper_limit and \
                under_limit < current_v_out < upper_limit:
            reward += 1.0
        else:
            reward -= 0.05

        if pre_v_out >= upper_limit and current_v_out >= upper_limit:
            if current_v_out < pre_v_out:
                reward += 0.01

        lim12 = 12 / self.normalize_ampere
        lim10 = 10 / self.normalize_ampere

        if prepre_iL > lim12 and pre_iL > lim12 and current_iL > lim12:
            reward -= 5.0
        elif prepre_iL > lim10 and pre_iL > lim10 and current_iL > lim10:
            reward -= 1.0

        return reward

    def __get_reward2(self):

        reward = 0
        error_limit_ratio = 0.02
        upper_limit2 = self.v_out_command * (1 + error_limit_ratio)
        under_limit2 = self.v_out_command * (1 - error_limit_ratio)

        error_limit_ratio = 0.005
        upper_limit05 = self.v_out_command * (1 + error_limit_ratio)
        under_limit05 = self.v_out_command * (1 - error_limit_ratio)

        prepre_iL, prepre_v_out = self.observation_list[self.time_index]
        pre_iL, pre_v_out = self.observation_list[self.time_index + 1]
        current_iL, current_v_out = self.observation_list[self.time_index + 2]

        if pre_v_out <= under_limit05 and current_v_out <= under_limit05:
            if current_v_out > pre_v_out:
                reward += 0.01

        if under_limit05 < prepre_v_out < upper_limit05 and \
                under_limit05 < pre_v_out < upper_limit05 and \
                under_limit05 < current_v_out < upper_limit05:
            reward += 1.0

            # if under_limit05 < prepre_v_out < upper_limit05 and\
            #    under_limit05 < pre_v_out < upper_limit05 and\
            #    under_limit05 < current_v_out < upper_limit05:
            #     reward += 1.0
            # elif under_limit05 < current_v_out < upper_limit05:
            #     reward += 0.2
        else:
            reward -= 0.05

        if pre_v_out >= upper_limit05 and current_v_out >= upper_limit05:
            if current_v_out < pre_v_out:
                reward += 0.01

        lim12 = 12 / self.normalize_ampere
        lim10 = 10 / self.normalize_ampere

        if prepre_iL > lim12 and pre_iL > lim12 and current_iL > lim12:
            reward -= 5.0
        elif prepre_iL > lim10 and pre_iL > lim10 and current_iL > lim10:
            reward -= 1.0

        return reward

    # 負荷の変化に必要な各種値を算出
    def set_load_value(self, load_value: float):
        if load_value <= 0:
            raise Exception("load value must be positive number")

        if self.during_load_change_flag:
            return False
        else:
            # 負荷変動割合が0なら無限大として計算し、すぐに負荷値を変更する
            if abs(self.resistance_change_rate) == 0:
                self.R_now = load_value
                return True

            # 以下負荷変動割合が自然数の場合
            self.during_load_change_flag = True
            # 負荷変動にかかる時間を計算
            self.load_transition_time = abs(load_value - self.R_ini) / self.resistance_change_rate
            # 負荷変動に要するstep数を計算
            self.load_transition_num_step = np.ceil(self.load_transition_time / self.dt_cont)
            # 負荷変動が増加か減少かで傾きの符号を変える
            if self.R_now > load_value:
                self.resistance_change_rate = -1 * abs(self.resistance_change_rate)
            else:
                self.resistance_change_rate = abs(self.resistance_change_rate)

            return True

    # 電圧の誤差についてのグラフを作成
    def save_v_error_graph(self, SAVE_PATH):
        time_list = [state[2] * 1000 for state in self.state_list]
        v_error_list = [abs(state[1] - self.v_out_command * self.E) for state in self.state_list]
        v_error_percentage_list = [100 * v_error / (self.v_out_command * self.E) for v_error in v_error_list]

        plt.figure()
        plt.plot(time_list, v_error_list)
        plt.xlabel("t [ms]")
        plt.ylabel("$v_{error}$ [V]")
        plt.savefig(f"{SAVE_PATH}/v_out_error.png", bbox_inches="tight", pad_inches=0.05, dpi=600)
        plt.close()

        plt.figure()
        plt.plot(time_list, v_error_percentage_list)
        plt.xlabel("t [ms]")
        plt.ylabel("$v_{error}$ [%]")
        plt.savefig(f"{SAVE_PATH}/v_out_error_percentage.png", bbox_inches="tight", pad_inches=0.05, dpi=600)
        plt.close()

    def __cal_circuit_with_sympy(self, sig):
        assert -1 <= sig <= 1

        iL_pre, v_out_pre, time_pre = self.state_list[self.time_index]

        # PWM波形のスイッチングタイミングを計算
        time_turn_on = ((self.CAREER_AMP - sig) / (2 * self.CAREER_AMP)) * (self.dt_cont * 0.5)
        time_turn_off = ((sig + self.CAREER_AMP) / (2 * self.CAREER_AMP)) * (self.dt_cont * 0.5) + self.dt_cont * 0.5

        # mode switch off
        C_ans1 = [self.C1_ufunc_off(0, iL_pre, v_out_pre, complex(self.R_now)),
                  self.C2_ufunc_off(0, iL_pre, v_out_pre, complex(self.R_now))]
        iL_now = self.iL_ufunc_off(time_turn_on, C_ans1[0], C_ans1[1], complex(self.R_now))
        v_out_now = self.v_out_ufunc_off(time_turn_on, C_ans1[0], C_ans1[1], complex(self.R_now))
        iL_now = complex(iL_now).real
        v_out_now = complex(v_out_now).real

        # mode switch on
        C_ans2 = [self.C1_ufunc_on(time_turn_on, iL_now, v_out_now, complex(self.R_now)),
                  self.C2_ufunc_on(time_turn_on, iL_now, v_out_now, complex(self.R_now))]
        iL_now = self.iL_ufunc_on(time_turn_off, C_ans2[0], C_ans2[1], complex(self.R_now))
        v_out_now = self.v_out_ufunc_on(time_turn_off, C_ans2[0], C_ans2[1], complex(self.R_now))
        iL_now = complex(iL_now).real
        v_out_now = complex(v_out_now).real

        # mode switch off
        C_ans3 = [self.C1_ufunc_off(time_turn_off, iL_now, v_out_now, complex(self.R_now)),
                  self.C2_ufunc_off(time_turn_off, iL_now, v_out_now, complex(self.R_now))]
        iL_now = self.iL_ufunc_off(self.dt_cont, C_ans3[0], C_ans3[1], complex(self.R_now))
        v_out_now = self.v_out_ufunc_off(self.dt_cont, C_ans3[0], C_ans3[1], complex(self.R_now))
        iL_now = complex(iL_now).real
        v_out_now = complex(v_out_now).real

        # smooth_scale が1より大きい場合のみ追加の計算を行う
        if self.smooth_flag:
            # 制御周期よりも細かく刻みで計算する
            ini_time = self.time_index * self.dt_cont
            dt = self.dt_cont / self.SMOOTH_SCALE
            for i in range(self.SMOOTH_SCALE):
                interim_time = dt * (i + 1)
                if interim_time < time_turn_on:
                    interim_iL = self.iL_ufunc_off(interim_time, C_ans1[0], C_ans1[1], complex(self.R_now))
                    interim_v_out = self.v_out_ufunc_off(interim_time, C_ans1[0], C_ans1[1], complex(self.R_now))
                elif interim_time < time_turn_off:
                    interim_iL = self.iL_ufunc_on(interim_time, C_ans2[0], C_ans2[1], complex(self.R_now))
                    interim_v_out = self.v_out_ufunc_on(interim_time, C_ans2[0], C_ans2[1], complex(self.R_now))
                else:
                    interim_iL = self.iL_ufunc_off(interim_time, C_ans3[0], C_ans3[1], complex(self.R_now))
                    interim_v_out = self.v_out_ufunc_off(interim_time, C_ans3[0], C_ans3[1], complex(self.R_now))
                # 実数に変換
                interim_iL = complex(interim_iL).real
                interim_v_out = complex(interim_v_out).real
                # リストに保存
                interim_state = np.array([interim_iL, interim_v_out, ini_time + interim_time], dtype=np.float32)
                self.state_list_smooth.append(interim_state)

        return iL_now, v_out_now

    def __cal_circuit_with_scipy(self, sig):
        assert -1 <= sig <= 1

        iL_now, v_out_now, time_pre = self.state_list[self.time_index]

        # PWM波形のスイッチングタイミングを計算
        time_turn_on = ((self.CAREER_AMP - sig) / (2 * self.CAREER_AMP)) * (self.dt_cont * 0.5)
        time_turn_off = ((sig + self.CAREER_AMP) / (2 * self.CAREER_AMP)) * (self.dt_cont * 0.5) + self.dt_cont * 0.5

        # 時間の刻みリストを作成
        t_eval_list_all = np.linspace(0, self.dt_cont)  # 一周期の期間を50分割したリストを作成

        iL_list_one_step = np.empty(0)
        v_out_list_one_step = np.empty(0)
        time_list_one_step = np.empty(0)

        # 微分方程式を定義
        def circuit_function_swith_off(t, y, L, C, R_ini, a):
            v = y[0]
            iL = y[1]

            dvdt = (1 / C) * (iL - v / (R_ini + a * t))
            diLdt = (-1 / L) * v

            return [dvdt, diLdt]

        def circuit_function_swith_on(t, y, L, C, R_ini, E, a):
            v = y[0]
            iL = y[1]

            dvdt = (1 / C) * (iL - v / (R_ini + a * t))
            diLdt = (1 / L) * (E - v)

            return [dvdt, diLdt]

        # 負荷変動に必要な残りステップ数
        remaining_steps = self.load_transition_num_step - self.load_transition_num_elapsed_step
        assert remaining_steps >= 1

        if remaining_steps >= 2:

            t_eval_list_all = np.hstack((t_eval_list_all, [time_turn_on, time_turn_off]))  # 作成したリストにスイッチングタイミングの時間を追加
            t_eval_list_all = np.sort(t_eval_list_all)  # ソートして昇順に並び替え
            t_eval_list_all = np.unique(t_eval_list_all)  # 重複する要素を削除
            t_turn_on_index = int((np.where(t_eval_list_all == time_turn_on))[-1])  # ソートし終えたリストでのスイッチングタイミングのindexを取得
            t_turn_off_index = int((np.where(t_eval_list_all == time_turn_off))[-1])
            t_list_1, t_list_2, t_list_3 = np.split(t_eval_list_all, [t_turn_on_index + 1, t_turn_off_index + 1])
            # それぞれのスイッチングタイミングまでのリストを分割することで取得

            if not time_turn_on == 0:
                t_ini = 0
                y_ini = np.array([v_out_now, iL_now]).reshape(-1)
                t_end = time_turn_on
                t_span = [t_ini, t_end]
                t_list = [t_end]    # もともとの記述
                ansivp = solve_ivp(circuit_function_swith_off, t_span, y_ini, t_eval=t_list_1,
                                   args=(self.L, self.C, self.R_now, self.resistance_change_rate),
                                   rtol=self.rtol, atol=self.atol)
                v_out_now = (ansivp.y)[0][-1]
                iL_now = (ansivp.y)[1][-1]

                # 途中の値を保存
                iL_list_one_step = np.hstack([iL_list_one_step, (ansivp.y)[1]])
                v_out_list_one_step = np.hstack([v_out_list_one_step, (ansivp.y)[0]])
                time_list_one_step = np.hstack([time_list_one_step, ansivp.t])

            if not time_turn_on == time_turn_off:
                t_ini = time_turn_on
                y_ini = np.array([v_out_now, iL_now]).reshape(-1)
                t_end = time_turn_off
                t_span = [t_ini, t_end]
                ansivp = solve_ivp(circuit_function_swith_on, t_span, y_ini, t_eval=t_list_2,
                                   args=(self.L, self.C, self.R_now, self.E, self.resistance_change_rate),
                                   rtol=self.rtol, atol=self.atol)
                v_out_now = (ansivp.y)[0][-1]
                iL_now = (ansivp.y)[1][-1]

                # 途中の値を保存
                iL_list_one_step = np.hstack([iL_list_one_step, (ansivp.y)[1]])
                v_out_list_one_step = np.hstack([v_out_list_one_step, (ansivp.y)[0]])
                time_list_one_step = np.hstack([time_list_one_step, ansivp.t])

            if not time_turn_off == self.dt_cont:
                t_ini = time_turn_off
                y_ini = np.array([v_out_now, iL_now]).reshape(-1)
                t_end = self.dt_cont
                t_span = [t_ini, t_end]
                ansivp = solve_ivp(circuit_function_swith_off, t_span, y_ini, t_eval=t_list_3,
                                   args=(self.L, self.C, self.R_now, self.resistance_change_rate),
                                   rtol=self.rtol, atol=self.atol)
                v_out_now = (ansivp.y)[0][-1]
                iL_now = (ansivp.y)[1][-1]

                # 途中の値を保存
                iL_list_one_step = np.hstack([iL_list_one_step, (ansivp.y)[1]])
                v_out_list_one_step = np.hstack([v_out_list_one_step, (ansivp.y)[0]])
                time_list_one_step = np.hstack([time_list_one_step, ansivp.t])

            # 途中の値をリストに記録
            for iL, v_out, time in zip(iL_list_one_step, v_out_list_one_step, time_list_one_step):
                self.state_list_smooth.append(np.array([iL, v_out,
                                                        time + self.time_index * self.dt_cont], dtype=np.float32))

            # 経過ステップ数をインクリメント
            self.load_transition_num_elapsed_step += 1

            # 現在の負荷の値を設定
            self.R_now += self.resistance_change_rate * self.dt_cont

        # ステップの途中で負荷変動が終わる場合
        else:
            remaining_time = self.load_transition_time - self.load_transition_num_elapsed_step * self.dt_cont
            assert 0 < remaining_time <= self.dt_cont

            t_eval_list_all = np.hstack((t_eval_list_all, [remaining_time, time_turn_on, time_turn_off]))
            t_eval_list_all = np.sort(t_eval_list_all)  # 昇順に並び替え
            t_eval_list_all = np.unique(t_eval_list_all)  # 重複する要素を削除

            t_turn_on_index = int((np.where(t_eval_list_all == time_turn_on))[-1])  # ソートし終えたリストでのスイッチングタイミングのindexを取得
            t_turn_off_index = int((np.where(t_eval_list_all == time_turn_off))[-1])
            t_switch_eq_index = int((np.where(t_eval_list_all == remaining_time))[-1])

            index_list = np.sort(np.array([t_turn_on_index, t_turn_off_index, t_switch_eq_index]))
            t_list_1, t_list_2, t_list_3, t_list_4 = np.split(t_eval_list_all, index_list + 1)

            # 条件1つ目
            if remaining_time <= time_turn_on:
                t_ini = 0
                y_ini = np.array([v_out_now, iL_now]).reshape(-1)
                t_end = remaining_time
                ansivp = solve_ivp(circuit_function_swith_off, [t_ini, t_end], y_ini, t_eval=t_list_1,
                                   args=(self.L, self.C, self.R_now, self.resistance_change_rate),
                                   rtol=self.rtol, atol=self.atol)
                v_out_now = (ansivp.y)[0][-1]
                iL_now = (ansivp.y)[1][-1]

                # 途中の値を保存
                iL_list_one_step = np.hstack([iL_list_one_step, (ansivp.y)[1]])
                v_out_list_one_step = np.hstack([v_out_list_one_step, (ansivp.y)[0]])
                time_list_one_step = np.hstack([time_list_one_step, ansivp.t])

                # 以下抵抗の値は固定になる
                self.R_now += self.resistance_change_rate * remaining_time

                if not remaining_time == time_turn_on:
                    t_ini = remaining_time
                    y_ini = np.array([v_out_now, iL_now]).reshape(-1)
                    t_end = time_turn_on
                    ansivp = solve_ivp(circuit_function_swith_off, [t_ini, t_end], y_ini, t_eval=t_list_2,
                                       args=(self.L, self.C, self.R_now, 0),
                                       rtol=self.rtol, atol=self.atol)
                    v_out_now = (ansivp.y)[0][-1]
                    iL_now = (ansivp.y)[1][-1]

                    # 途中の値を保存
                    iL_list_one_step = np.hstack([iL_list_one_step, (ansivp.y)[1]])
                    v_out_list_one_step = np.hstack([v_out_list_one_step, (ansivp.y)[0]])
                    time_list_one_step = np.hstack([time_list_one_step, ansivp.t])

                if not time_turn_on == time_turn_off:
                    t_ini = time_turn_on
                    y_ini = np.array([v_out_now, iL_now]).reshape(-1)
                    t_end = time_turn_off
                    ansivp = solve_ivp(circuit_function_swith_on, [t_ini, t_end], y_ini, t_eval=t_list_3,
                                       args=(self.L, self.C, self.R_now, self.E, 0),
                                       rtol=self.rtol, atol=self.atol)
                    v_out_now = (ansivp.y)[0][-1]
                    iL_now = (ansivp.y)[1][-1]

                    # 途中の値を保存
                    iL_list_one_step = np.hstack([iL_list_one_step, (ansivp.y)[1]])
                    v_out_list_one_step = np.hstack([v_out_list_one_step, (ansivp.y)[0]])
                    time_list_one_step = np.hstack([time_list_one_step, ansivp.t])

                if not time_turn_off == self.dt_cont:
                    t_ini = time_turn_off
                    y_ini = np.array([v_out_now, iL_now]).reshape(-1)
                    t_end = self.dt_cont
                    ansivp = solve_ivp(circuit_function_swith_off, [t_ini, t_end], y_ini, t_eval=t_list_4,
                                       args=(self.L, self.C, self.R_now, 0),
                                       rtol=self.rtol, atol=self.atol)
                    v_out_now = (ansivp.y)[0][-1]
                    iL_now = (ansivp.y)[1][-1]

                    # 途中の値を保存
                    iL_list_one_step = np.hstack([iL_list_one_step, (ansivp.y)[1]])
                    v_out_list_one_step = np.hstack([v_out_list_one_step, (ansivp.y)[0]])
                    time_list_one_step = np.hstack([time_list_one_step, ansivp.t])

            # 条件2つ目
            elif remaining_time <= time_turn_off:

                if not time_turn_on == 0:
                    t_ini = 0
                    y_ini = np.array([v_out_now, iL_now]).reshape(-1)
                    t_end = time_turn_on
                    ansivp = solve_ivp(circuit_function_swith_off, [t_ini, t_end], y_ini, t_eval=t_list_1,
                                       args=(self.L, self.C, self.R_now, self.resistance_change_rate),
                                       rtol=self.rtol, atol=self.atol)
                    v_out_now = (ansivp.y)[0][-1]
                    iL_now = (ansivp.y)[1][-1]

                    # 途中の値を保存
                    iL_list_one_step = np.hstack([iL_list_one_step, (ansivp.y)[1]])
                    v_out_list_one_step = np.hstack([v_out_list_one_step, (ansivp.y)[0]])
                    time_list_one_step = np.hstack([time_list_one_step, ansivp.t])

                t_ini = time_turn_on
                y_ini = np.array([v_out_now, iL_now]).reshape(-1)
                t_end = remaining_time
                ansivp = solve_ivp(circuit_function_swith_on, [t_ini, t_end], y_ini, t_eval=t_list_2,
                                   args=(self.L, self.C, self.R_now, self.E, self.resistance_change_rate),
                                   rtol=self.rtol, atol=self.atol)
                v_out_now = (ansivp.y)[0][-1]
                iL_now = (ansivp.y)[1][-1]

                # 途中の値を保存
                iL_list_one_step = np.hstack([iL_list_one_step, (ansivp.y)[1]])
                v_out_list_one_step = np.hstack([v_out_list_one_step, (ansivp.y)[0]])
                time_list_one_step = np.hstack([time_list_one_step, ansivp.t])

                # 以下抵抗の値は固定になる
                self.R_now += self.resistance_change_rate * remaining_time

                if not remaining_time == time_turn_off:
                    t_ini = remaining_time
                    y_ini = np.array([v_out_now, iL_now]).reshape(-1)
                    t_end = time_turn_off
                    ansivp = solve_ivp(circuit_function_swith_on, [t_ini, t_end], y_ini, t_eval=t_list_3,
                                       args=(self.L, self.C, self.R_now, self.E, 0),
                                       rtol=self.rtol, atol=self.atol)
                    v_out_now = (ansivp.y)[0][-1]
                    iL_now = (ansivp.y)[1][-1]

                    # 途中の値を保存
                    iL_list_one_step = np.hstack([iL_list_one_step, (ansivp.y)[1]])
                    v_out_list_one_step = np.hstack([v_out_list_one_step, (ansivp.y)[0]])
                    time_list_one_step = np.hstack([time_list_one_step, ansivp.t])

                if not time_turn_off == self.dt_cont:
                    t_ini = time_turn_off
                    y_ini = np.array([v_out_now, iL_now]).reshape(-1)
                    t_end = self.dt_cont
                    ansivp = solve_ivp(circuit_function_swith_off, [t_ini, t_end], y_ini, t_eval=t_list_4,
                                       args=(self.L, self.C, self.R_now, 0),
                                       rtol=self.rtol, atol=self.atol)
                    v_out_now = (ansivp.y)[0][-1]
                    iL_now = (ansivp.y)[1][-1]

                    # 途中の値を保存
                    iL_list_one_step = np.hstack([iL_list_one_step, (ansivp.y)[1]])
                    v_out_list_one_step = np.hstack([v_out_list_one_step, (ansivp.y)[0]])
                    time_list_one_step = np.hstack([time_list_one_step, ansivp.t])

            # 条件3つ目
            else:
                assert time_turn_off < remaining_time <= self.dt_cont

                if not time_turn_on == 0:
                    t_ini = 0
                    y_ini = np.array([v_out_now, iL_now]).reshape(-1)
                    t_end = time_turn_on
                    ansivp = solve_ivp(circuit_function_swith_off, [t_ini, t_end], y_ini, t_eval=t_list_1,
                                       args=(self.L, self.C, self.R_now, self.resistance_change_rate),
                                       rtol=self.rtol, atol=self.atol)
                    v_out_now = (ansivp.y)[0][-1]
                    iL_now = (ansivp.y)[1][-1]

                    # 途中の値を保存
                    iL_list_one_step = np.hstack([iL_list_one_step, (ansivp.y)[1]])
                    v_out_list_one_step = np.hstack([v_out_list_one_step, (ansivp.y)[0]])
                    time_list_one_step = np.hstack([time_list_one_step, ansivp.t])

                if not time_turn_on == time_turn_off:
                    t_ini = time_turn_on
                    y_ini = np.array([v_out_now, iL_now]).reshape(-1)
                    t_end = time_turn_off
                    ansivp = solve_ivp(circuit_function_swith_on, [t_ini, t_end], y_ini, t_eval=t_list_2,
                                       args=(self.L, self.C, self.R_now, self.E, self.resistance_change_rate),
                                       rtol=self.rtol, atol=self.atol)
                    v_out_now = (ansivp.y)[0][-1]
                    iL_now = (ansivp.y)[1][-1]

                    # 途中の値を保存
                    iL_list_one_step = np.hstack([iL_list_one_step, (ansivp.y)[1]])
                    v_out_list_one_step = np.hstack([v_out_list_one_step, (ansivp.y)[0]])
                    time_list_one_step = np.hstack([time_list_one_step, ansivp.t])

                t_ini = time_turn_off
                y_ini = np.array([v_out_now, iL_now]).reshape(-1)
                t_end = remaining_time
                ansivp = solve_ivp(circuit_function_swith_off, [t_ini, t_end], y_ini, t_eval=t_list_3,
                                   args=(self.L, self.C, self.R_now, self.resistance_change_rate),
                                   rtol=self.rtol, atol=self.atol)
                v_out_now = (ansivp.y)[0][-1]
                iL_now = (ansivp.y)[1][-1]

                # 途中の値を保存
                iL_list_one_step = np.hstack([iL_list_one_step, (ansivp.y)[1]])
                v_out_list_one_step = np.hstack([v_out_list_one_step, (ansivp.y)[0]])
                time_list_one_step = np.hstack([time_list_one_step, ansivp.t])

                # 以下抵抗の値は固定になる
                self.R_now += self.resistance_change_rate * remaining_time

                if not remaining_time == self.dt_cont:
                    t_ini = remaining_time
                    y_ini = np.array([v_out_now, iL_now]).reshape(-1)
                    t_end = self.dt_cont
                    ansivp = solve_ivp(circuit_function_swith_off, [t_ini, t_end], y_ini, t_eval=t_list_4,
                                       args=(self.L, self.C, self.R_now, 0),
                                       rtol=self.rtol, atol=self.atol)
                    v_out_now = (ansivp.y)[0][-1]
                    iL_now = (ansivp.y)[1][-1]

                    # 途中の値を保存
                    iL_list_one_step = np.hstack([iL_list_one_step, (ansivp.y)[1]])
                    v_out_list_one_step = np.hstack([v_out_list_one_step, (ansivp.y)[0]])
                    time_list_one_step = np.hstack([time_list_one_step, ansivp.t])

            # 途中の値をリストに記録
            for iL, v_out, time in zip(iL_list_one_step, v_out_list_one_step, time_list_one_step):
                self.state_list_smooth.append(np.array([iL, v_out,
                                                        time + self.time_index * self.dt_cont], dtype=np.float32))

            # 負荷変動中フラグを変更
            self.during_load_change_flag = False


        return iL_now, v_out_now

    def draw_load_value_transition(self):

        plt.figure()
        plt.plot(range(len(self.load_value_list)), self.load_value_list)
        # plt.ylim(0, 25)
        plt.xlabel("step")
        plt.ylabel("load value [ohm]")