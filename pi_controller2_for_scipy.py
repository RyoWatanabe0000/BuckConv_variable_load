import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

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
MAX_EPI_LEN = int(TIME_LIMIT / TIME_contorl)       #ループの回数上限

# 素子値
E = 100
L = 0.50e-3
R_ini = 25
R_now = R_ini
C = 220e-6

iR_INI = 0.0  #抵抗に流れる電流の初期値
iL_INI = 0.0  #インダクタに流れる電流の初期値
V_REF = 25  # 電圧のリファレンス値
CLIP_AMPERE_VALUE = 15
now_step = 0    #現在のステップカウント
# 電圧PI制御のゲイン
KI_v = 0.05
KP_v = 2
# 電流PI制御のゲイン
KI_i = 0.005
KP_i = 0.5
i_err_integ = 0 #電流誤差の積分値保存用
v_err_integ = 0 #電圧誤差の積分値保存用
state_list = []
sig_list = []
# voltage_list = []
DATA_SAVE_DIR = "data"

circuit = BuckConverterEnv(dt=TIME_contorl, E_ini=E, R_ini=R_ini, L_ini=L, C_ini=C,
                           career_amp=1, smooth_scale=50, clip_ampere_value=CLIP_AMPERE_VALUE)
observation = circuit.reset(v_out_command=V_REF, resistance_change_rate=10000, v_out_ini=25, il_ini=1.0)

for i in range(MAX_EPI_LEN):
    if i == 50:
        R_now = 5
        circuit.set_load_value(R_now)

    iL = observation[0] * CLIP_AMPERE_VALUE
    v_out = observation[1] * E

    # 電圧のPI制御
    v_err = V_REF - v_out
    v_err_integ += v_err * KI_v
    iLc = KP_v * v_err + v_err_integ

    # iLcを-10から10の間でクリップ
    iLc = 13 if iLc > 13 else iLc
    iLc = -13 if iLc < -13 else iLc

    # 電流のPI制御
    i_err = iLc - iL
    i_err_integ += i_err * KI_i
    sig = KP_i * i_err + i_err_integ

    # sig = 0.5
    sig = np.array([sig], dtype=np.float32)
    # sig = np.clip(sig, a_min=-1, a_max=1)
    sig_list.append(sig)
    observation, _, _, _ = circuit.step(sig)

fig, ax1, ax2 = circuit.render_smooth()
# fig, ax1, ax2 = circuit.render()
# ax1.set_xlim(6, 8)
# ax2.set_xlim(6, 8)
# circuit.draw_load_value_transition()
# plt.ylim(24, 27)
# plt.xlim(1.5, 3.0)
# plt.figure()
# plt.plot(range(len(voltage_list)), voltage_list)
plt.show()

# 波形データをcsvとして保存
# state_list = circuit.get_smooth_status_list()
# iL_list = [temp[0] for temp in state_list]
# v_out_list = [temp[1] for temp in state_list]
# time_list = [temp[2] for temp in state_list]
# df = pd.DataFrame(list(zip(time_list, iL_list, v_out_list)), columns=["time[s]", "iL[A]", "v out[V]"], index=None)
# df.to_csv(f"{DATA_SAVE_DIR}/current_data_volgain_{KP_v}-{KI_v}_curgain_{KP_i}-{KI_i}.csv", index=False)