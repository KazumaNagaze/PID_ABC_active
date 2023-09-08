import numpy as np
from scipy.integrate import quad
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.animation import FuncAnimation
import math
import scipy.optimize as opt
import control as ctrl
from tkinter import ttk
import tkinter as tk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# PIDパラメータの初期値
Kp = 1.0
Ki = 1.0
Kd = 1.0

# 目標値（r(t)）と制御系の応答（y(t)）をシミュレーションまたは実験から取得
def target_value(t):
    # 目標値の関数を定義
    # 例: 1.0（一定の目標値）
    return 1.0

# 目的関数の定義
def objective_function(Kp, Ki, Kd, t):
    # シミュレーション時間の範囲を設定（例: 0から10秒まで）
    t_min = 0
    t_max = 10

    # 目的関数の積分を計算
    result, _ = quad(lambda t: (target_value(t) - simulate())**2, t_min, t_max)
    return result

# 評価関数を定義
def fitness_function(Kp, Ki, Kd, t):
    # 目的関数の値を計算
    J_value = objective_function(Kp, Ki, Kd, t)
    
    # 評価関数を計算
    fitness_value = 1.0 / (1.0 + J_value)
    
    return fitness_value

# ABCアルゴリズムで最適なPIDパラメータを推定する部分
#３Dグラフの定義
fig = plt.figure(figsize=(6, 6))
ax = fig.add_subplot(111, projection='3d')
ax.set_xlabel('Kp')
ax.set_xlabel('Ki')
ax.set_xlabel('Kd')
K_p = np.arange(0, 10)
K_i = np.arange(0, 10)
K_d = np.arange(0, 10)
X, Y, Z = np.meshgrid(K_p, K_i, K_d)
 
# 初期設定
N = 100 # 働き蜂と傍観蜂の個体数
d = 3 # 次元
s = np.zeros(N) #更新カウント
lim = 30
xmax  = 500
xmin =0
G = 300 # 繰り返す回数
 
x_best = [0,0,0] #x_bestの初期化

x = np.zeros((N,d)) #蜂の配置
for i in range(N):
    x[i] = (xmax-xmin)*np.random.rand(d) + xmin
 
# ルーレット選択用関数
def roulette_choice(w):
    tot = []
    acc = 0
    for e in w:
        acc += e
        tot.append(acc)
 
    r = np.random.random() * acc
    for i, e in enumerate(tot):
        if r <= e:
            return i

 
# 繰り返し
best_value = []
x_best_value = [0, 0, 0]
ims = []
for g in range(G):
    # 働き蜂employee bee step
    for i in range(N):
        v = x.copy()
        k = i
        while k == i:
            k = np.random.randint(N)
        j = np.random.randint(d)
        r = np.random.rand()*2 - 1 # -1から1までの一様乱数
        v[i,j] = x[i,j] + r * (x[i,j] - x[k,j])
 
        if fitness_function(*x[i], g) < fitness_function(*v[i], g): #適合度に基づいて働き蜂の情報を更新
            x[i] = v[i]
        if fitness_function(*x[i], g) <= fitness_function(*v[i], g):
            s[i] = 0
        else: s[i] += 1
 
    # 傍観蜂onlooker bee step
    w = []
    for i in range(N):
        w.append(fitness_function(*x[i], g))
    
    for l in range(N):
        i = roulette_choice(w)
        j = np.random.randint(d)
        r = np.random.rand()*2 - 1 # -1から1までの一様乱数
        v[i,j] = x[i,j] + r * (x[i,j] - x[k,j])
        if fitness_function(*x[i], g) < fitness_function(*v[i], g):
            x[i] = v[i]
        if fitness_function(*x[i], g) <= fitness_function(*v[i], g):
            s[i] = 0
        else: s[i] += 1
 
 
    # 斥候蜂scout bee step
    for i in range(N):
        if s[i] >= lim:
            for j in range(d):
                x[i,j] = np.random.rand()*(xmax-xmin) + xmin
            s[i] = 0
 
    # 最良個体の発見
    z = np.zeros(N)
    best = float('-inf')  # 初期値をマイナス無限大に設定
    for i in range(N):
    #    if best > objective_function(*x[i], g): #時変関数に対応させるために削除
        z[i] = fitness_function(*x[i], g)
        if z[i] > best:
            best = max(z)
            x_best = x[i].copy()
    
    best_value.append(objective_function(*x_best, g))
    x_best_value.append(x_best)
    print("発見した最適解：{}\nそのときのPIDパラメータ：{}".format(objective_function(*x_best, g), x_best))

    #アニメーションの座標保存
    im = ax.scatter3D(*x_best, c='r', s=0.5, alpha=0.5)
    ims.append(im)

 
# 結果の表示
# 3D平面上で最適解の座標をプロット
ax.plot_wireframe(*best_value, color='b', linewidth=0.3, alpha=0.3)   
ani = animation.ArtistAnimation(fig, ims)
plt.show()
#ani.save('./3D-animation.gif', writer='pillow')
#plt.close()
'''
#最適値をプロット
plt.plot(range(G), best_value)
plt.yscale('log')
plt.title("Time-varying funktion")
plt.xlabel("試行回数")
plt.ylabel("発見した最小値")
plt.show()
'''
# PIDパラメータを設定する関数
def set_pid_parameters(x, y, z):
    global Kp, Ki, Kd
    Kp = float(x.get())
    Ki = float(y.get())
    Kd = float(z.get())

# シミュレーションを実行する関数
def simulate():
    set_pid_parameters()

    # シミュレーション時間
    time = np.linspace(0, 10, 1000)

    # 制御対象のシステムモデル（二次遅れ系）
    omega_n = 2.0      # 自然角周波数
    zeta = 1.5         # 減衰比（1より大きい）
    system = ctrl.TransferFunction([omega_n**2], [1, 2*zeta*omega_n, omega_n**2])

    # PID制御器
    controller = ctrl.TransferFunction([Kd, Ki, Kp], [1, 0])

    # 制御系を組み立て
    closed_loop_system = ctrl.feedback(controller * system, 1)

    # ステップ入力信号
    input_signal = np.ones_like(time)

    # 制御系の応答を計算
    time, response = ctrl.step_response(closed_loop_system, time)

    # プロット
    plt.figure()
    plt.plot(time, input_signal, label="目標値")
    plt.plot(time, response, label="応答")
    plt.xlabel("時間")
    plt.ylabel("値")
    plt.legend()
    plt.grid(True)
    plt.show()

# Tkinterウィンドウの設定
root = tk.Tk()
root.title("PID-ABC Control (Second-Order System)")

# PIDパラメータ入力フレーム
pid_frame = ttk.LabelFrame(root, text="PIDパラメータ")
pid_frame.grid(row=0, column=0, padx=10, pady=10)

Kp_label = ttk.Label(pid_frame, text="Kp:")
Kp_label.grid(row=0, column=0, padx=5, pady=5)
Kp_scale = tk.Scale(pid_frame, from_=0.0, to=10.0, resolution=0.01, orient="horizontal")
Kp_scale.grid(row=0, column=1, padx=5, pady=5)
Kp_scale.set(Kp)

Ki_label = ttk.Label(pid_frame, text="Ki:")
Ki_label.grid(row=1, column=0, padx=5, pady=5)
Ki_scale = tk.Scale(pid_frame, from_=0.0, to=10.0, resolution=0.01, orient="horizontal")
Ki_scale.grid(row=1, column=1, padx=5, pady=5)
Ki_scale.set(Ki)

Kd_label = ttk.Label(pid_frame, text="Kd:")
Kd_label.grid(row=2, column=0, padx=5, pady=5)
Kd_scale = tk.Scale(pid_frame, from_=0.0, to=10.0, resolution=0.01, orient="horizontal")
Kd_scale.grid(row=2, column=1, padx=5, pady=5)
Kd_scale.set(Kd)

update_button = ttk.Button(pid_frame, text="パラメータ更新", command=set_pid_parameters(*x_best))
update_button.grid(row=3, column=0, columnspan=2, padx=5, pady=5)

# シミュレーションボタン
simulate_button = ttk.Button(root, text="シミュレーション実行", command=simulate(*x_best))
simulate_button.grid(row=1, column=0, padx=10, pady=10)

root.mainloop()
