from scipy.integrate import solve_ivp
import numpy as np
import matplotlib.pyplot as plt
import sympy as sp
import time as tm

def circuit_swith_off(t, y, L, C, R):
    v = y[0]
    iL = y[1]

    dvdt = (1/C) * (iL - v/R)
    diLdt = (-1/L) * v

    return [dvdt, diLdt]

def circuit_swith_on(t, y, L, C, R, E):
    v = y[0]
    iL = y[1]

    dvdt = (1/C) * (iL - v/R)
    diLdt = (1/L) * (E - v)

    return [dvdt, diLdt]


t_ini = 0
y_ini = [0, 0]
t_end = 0.004
t_span = [t_ini, t_end]

t_list = np.linspace(t_ini, t_end, 201)
# t_list = [t_end]

L_value = 0.35e-3; C_value = 0.22e-3; R_value = 10; E_value = 100
# ansivp = solve_ivp(circuit_swith_off, t_span, y_ini, t_eval=t_list, args=(L, C, R), rtol=1e-12, atol=1e-14)
start_time1 = tm.time()

ansivp = solve_ivp(circuit_swith_on, t_span, y_ini, t_eval=t_list, args=(L_value, C_value, R_value, E_value),
                   rtol=1e-6, atol=1e-8)
#default rtol -12 atol -14
elapsed_time1 = tm.time() - start_time1

# シンボルの定義
t = sp.symbols('t', real=True)
iL_ini, iR_ini, Q_ini, v_out_ini = sp.symbols('iL_ini iR_ini Q_ini v_out_ini', real=True)
C1, C2 = sp.symbols('C1 C2')
R, E, C, L = sp.symbols('R E C L')
Q = sp.Function('Q')(t)

# SW off
# コンデンサに蓄えられる電荷Qに関する二階線形微分方程式を解く
ode_Q_off = sp.Eq(sp.diff(Q, t, 2) + (1 / (C * R)) * sp.diff(Q, t, 1) + (1 / (L * C)) * Q, 0)
Q_func_off = sp.dsolve(ode_Q_off, hint='nth_linear_constant_coeff_homogeneous')
v_out_func_off = Q_func_off.rhs / C
iL_func_off = (1 / (C * R)) * Q_func_off.rhs + sp.diff(Q_func_off.rhs, t, 1)

# 定数C1C2についての方程式を立式
C_func_off = sp.solve([sp.Eq(v_out_func_off, v_out_ini), sp.Eq(iL_func_off, iL_ini)], [C1, C2], hint="best")

# SW on
# コンデンサに蓄えられる電荷Qに関する二階線形微分方程式を解く
ode_Q_on = sp.Eq(sp.diff(Q, t, 2) + (1 / (C * R)) * sp.diff(Q, t, 1) + (1 / (L * C)) * Q, E / L)
Q_func_on = sp.dsolve(ode_Q_on, hint='nth_linear_constant_coeff_undetermined_coefficients')
v_out_func_on = Q_func_on.rhs / C
iL_func_on = Q_func_on.rhs / (C * R) + sp.diff(Q_func_off.rhs, t, 1)

# 定数C1C2についての方程式を立式
C_func_on = sp.solve([sp.Eq(v_out_func_on, v_out_ini), sp.Eq(iL_func_on, iL_ini)], [C1, C2], hint="best")
v_out_func_off = v_out_func_off.subs([(L, L_value), (C, C_value), (E, E_value)])
v_out_func_on = v_out_func_on.subs([(L, L_value), (C, C_value), (E, E_value)])
iL_func_off = iL_func_off.subs([(L, L_value), (C, C_value), (E, E_value)])
iL_func_on = iL_func_on.subs([(L, L_value), (C, C_value), (E, E_value)])
C_func_off[C1] = C_func_off[C1].subs([(L, L_value), (C, C_value), (E, E_value)])
C_func_off[C2] = C_func_off[C2].subs([(L, L_value), (C, C_value), (E, E_value)])
C_func_on[C1] = C_func_on[C1].subs([(L, L_value), (C, C_value), (E, E_value)])
C_func_on[C2] = C_func_on[C2].subs([(L, L_value), (C, C_value), (E, E_value)])

# 電流と任意定数Cの関数を扱いやすい形に変換(numpyのufuncに変換)
v_out_ufunc_off = sp.lambdify((t, C1, C2, R), v_out_func_off, "numpy")
iL_ufunc_off = sp.lambdify((t, C1, C2, R), iL_func_off, "numpy")
v_out_ufunc_on = sp.lambdify((t, C1, C2, R), v_out_func_on, "numpy")
iL_ufunc_on = sp.lambdify((t, C1, C2, R), iL_func_on, "numpy")

C1_ufunc_off = sp.lambdify((t, iL_ini, v_out_ini, R), C_func_off[C1], "numpy")
C1_ufunc_on = sp.lambdify((t, iL_ini, v_out_ini, R), C_func_on[C1], "numpy")
C2_ufunc_off = sp.lambdify((t, iL_ini, v_out_ini, R), C_func_off[C2], "numpy")
C2_ufunc_on = sp.lambdify((t, iL_ini, v_out_ini, R), C_func_on[C2], "numpy")


v_list = []
iL_list = []
# mode switch on
C_ans2 = [C1_ufunc_on(0, 0, 0, complex(R_value)),
          C2_ufunc_on(0, 0, 0, complex(R_value))]
start_time2 = tm.time()
for t in t_list:
    iL = iL_ufunc_on(t, C_ans2[0], C_ans2[1], complex(R_value))
    v_out = v_out_ufunc_on(t, C_ans2[0], C_ans2[1], complex(R_value))
    iL_list.append(complex(iL).real)
    v_list.append(complex(v_out).real)
elapsed_time2 = tm.time() - start_time2


ans_list = list(ansivp.y)
plt.figure()
plt.scatter(t_list, (ansivp.y)[0], s=10, marker="o")
plt.plot(t_list, v_list, linewidth=0.5)
plt.show()

plt.figure()
plt.scatter(t_list, (ansivp.y)[1], s=10, marker="o")
plt.plot(t_list, iL_list, linewidth=1.2)
plt.show()

print(elapsed_time1)
print(elapsed_time2)

# print((ansivp.y)[0])
# print((ansivp.y)[1])

# print(np.max(np.abs((ansivp.y)[0] - np.array(v_list))))
# print(np.min(np.abs((ansivp.y)[0] - np.array(v_list))))