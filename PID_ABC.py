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


# 時間に応じて変化する目標値
def target_value(t):
    return  3 + 0.5 * np.sin(0.05 * t)

'''
# 時間に応じて変化する外乱
def disturbance(t):
    return 0.3 * np.sin(0.2 * t)
'''

# 評価の定義
def objective_function(Kp, Ki, Kd, t):
    if not hasattr(objective_function, "initialized"):
        # 初回の呼び出しの場合
        integral = 0  # 積分項のリセット
        prev_error = 0  # 前回のエラーのリセット
        setattr(objective_function, "initialized", True)  # 初回のみ初期化フラグを設定
    else:
        # 2回目以降の呼び出しで、tが変化した場合
        if t != objective_function.prev_t:
            integral = 0  # 積分項のリセット
            prev_error = 0  # 前回のエラーのリセット
            objective_function.prev_t = t  # tを更新
        else:
            integral = getattr(objective_function, "integral", 0)  # 初回以外は保存されたintegralを使用
            prev_error = getattr(objective_function, "prev_error", 0)  # 初回以外は保存されたprev_errorを使用

    # ここに通常の処理を追加する
    error = target_value(t) - output  # 現在値と目標値とのエラーを計算
    integral += error  # 積分項
    derivative = error - prev_error  # 微分項
    result = Kp * error + Ki * integral + Kd * derivative  # PID controller出力を計算

    # 前回のエラーとintegralを保存
    setattr(objective_function, "prev_error", error)
    setattr(objective_function, "integral", integral)
    objective_function.prev_t = t  # 前回のtを保存

    return result


# 評価関数を定義
def fitness_function(Kp, Ki, Kd, t):
    # 目的関数の値を計算
    J_value = objective_function(Kp, Ki, Kd, t)
    
    # 評価関数を計算
    fitness_value = 1.0 / (1.0 + J_value)
    
    return fitness_value

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

# ABCアルゴリズムで最適なPIDパラメータを推定する関数
def ABC_algo(t):
       
    # 初期設定
    N = 100 # 働き蜂と傍観蜂の個体数
    d = 3 # 次元
    s = np.zeros(N) #更新カウント
    lim = 30
    xmax  = 30
    xmin =0    
    x_best = [0,0,0] #x_bestの初期化

    x = np.zeros((N,d)) #蜂の配置
    for i in range(N):
        x[i] = (xmax-xmin)*np.random.rand(d) + xmin
    
    # 繰り返し
    best_value = []
    x_best_value = [0, 0, 0]
    ims = []

    # 働き蜂employee bee step
    for i in range(N):
        v = x.copy()
        k = i
        while k == i:
            k = np.random.randint(N)
        j = np.random.randint(d)
        r = np.random.rand()*2 - 1 # -1から1までの一様乱数
        v[i,j] = x[i,j] + r * (x[i,j] - x[k,j])

        if fitness_function(*x[i], t) < fitness_function(*v[i], t): #適合度に基づいて働き蜂の情報を更新
            x[i] = v[i]
        if fitness_function(*x[i], t) <= fitness_function(*v[i], t):
            s[i] = 0
        else: s[i] += 1

    # 傍観蜂onlooker bee step
    w = []
    for i in range(N):
        w.append(fitness_function(*x[i], t))
    
    for l in range(N):
        i = roulette_choice(w)
        j = np.random.randint(d)
        r = np.random.rand()*2 - 1 # -1から1までの一様乱数
        v[i,j] = x[i,j] + r * (x[i,j] - x[k,j])
        if fitness_function(*x[i], t) < fitness_function(*v[i], t):
            x[i] = v[i]
        if fitness_function(*x[i], t) <= fitness_function(*v[i], t):
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
        z[i] = fitness_function(*x[i], t)
        if z[i] > best:
            best = max(z)
            x_best = x[i].copy()
    
#       best_value.append(objective_function(*x_best, t))
    x_best_value.append(x_best)
    print("発見した最適解：{}\nそのときのPIDパラメータ：{}".format(objective_function(*x_best, t), x_best))
    return x_best

# シミュレーションを実行する関数
def simulate(Kp, Ki, Kd, t):

    # 制御対象のシステムモデル（二次遅れ系）
    omega_n = 2.0 + 1.0 * np.sin(0.05 * t)      # 自然角周波数
    damping_ratio = 0.2 + 0.1 * np.sin(0.1 * t)         # 減衰比
    system = ctrl.TransferFunction([omega_n**2], [2 * damping_ratio * omega_n])

    # PID制御器
    controller = ctrl.TransferFunction([Kd, Ki, Kp], [1, 0])

    # 制御系を組み立て
    closed_loop_system = ctrl.feedback(controller * system, 1)

    # ステップ入力信号
    input_signal = np.ones_like(t)

    # 制御系の応答を計算
    time, response = ctrl.step_response(closed_loop_system, t)

    return response

#main
# シミュレーション時間
time = np.linspace(0, 10, 1000)
global output
output = np.zeros(len(time))
PID = [0, 0 ,0]
Kp_value = []
Ki_value = []
Kd_value = []
for t in time:
    PID = ABC_algo(t)
    Kp_value.append(PID[0])
    Ki_value.append(PID[1])
    Kd_value.append(PID[2])
    y = simulate(*PID, t)
    output.append(y)

