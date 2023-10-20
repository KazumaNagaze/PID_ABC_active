import numpy as np
import control as ctrl
import matplotlib.pyplot as plt

# グローバル変数の初期化
error = 0  # 目標値との誤差を保存する変数
overshoot = 0 #オーバーシュートを保存する変数
settling_time = 0  # 目標値に収束する時間を保存する変数

# 評価関数のパラメータ（重要度の調整）
error_weight = 2.0   # 目標値との誤差の重要度
settling_time_weight = 1.5   # 目標収束時間の重要度
overshoot_weight = 1.5  # オーバーシュートの重要度

# 制約範囲の設定
Kp_min = 0
Kp_max = 10.0
Ki_min = 0
Ki_max = 10.0
Kd_min = 0
Kd_max = 10.0

# PIDの定義
def objective_function(Kp, Ki, Kd):
    global integral_sum
    global error
    global settling_time
    global overshoot 

    # シミュレーション用時間設定
    time = np.linspace(0, 10, 1000)

    # 制御対象の伝達関数を定義 (例: 2次遅れ系)
    num = [2]
    den = [1, 0.6, 1]

    # 制御対象の伝達関数をシステムとして定義
    system = ctrl.TransferFunction(num, den)

    # 制御器を構築
    controller = ctrl.TransferFunction([Kd, Kp, Ki], [1, 0])
    closed_loop_system = ctrl.feedback(system * controller, 1, -1)

    feedback_connection = ctrl.TransferFunction([2*Kd, 2*Kp, 2*Ki], [1, 2*Kd+0.6, 2*Kp+1, 2*Ki])

    # 制御系の応答を計算
    _, response = ctrl.step_response(feedback_connection, time)

    # 目標値に収束する時間を計算
    settling_time = calculate_settling_time(time, response)

    # オーバーシュートを計算
    peak_value = np.max(response)
    if peak_value > 1.0:
        overshoot = (peak_value - 1.0) / 1.0  # オーバーシュートを計算（目標値が1.0の場合）
    else:
        overshoot = 0.0  # オーバーシュートがない場合は0.0に設定

    return response


def calculate_settling_time(time, response):
    # 目標値に収束する時間を計算する関数を実装
    # 収束条件（通常は±5%の範囲内に入る時間）を設定します

    global settling_time
    settling_percentage = 0.05  # 通常は±5%の範囲内に入る時間を指定
    target_value = 1.0  # 目標値（この値を目指す）

    for i, y in enumerate(response):
        if abs(y - target_value) <= settling_percentage * target_value:
            settling_time = time[i]
            break

    return settling_time

# 評価関数を定義
def fitness_function(Kp, Ki, Kd):
    global error
    global settling_time
    global overshoot 

    convergence_value = objective_function(Kp, Ki, Kd)

    # 目標値との誤差を計算
    error = np.abs(convergence_value[-1] - 1.0)

    # 評価値の計算（各項目に対する重み付けを考慮）
    error_evaluation = error_weight * (1 / (1 + error))
    settling_time_evaluation = settling_time_weight * (1 / (1 + settling_time))
    overshoot_evaluation = overshoot_weight * (1 / (1 + overshoot))

    # 評価を合計して最終評価を計算
    total_evaluation = (
        error_evaluation
        + settling_time_evaluation
        + overshoot_evaluation
    )
    
    return total_evaluation
    


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
def ABC_algo():    
    # 初期設定
    N = 100 # 働き蜂と傍観蜂の個体数
    d = 3 # 次元
    s = np.zeros(N) #更新カウント
    lim = 50
    #xmax  = 10
    #xmin = 0    
    G = 100 # 繰り返す回数
    x_best = [0,0,0] 

    # ループごとの評価関数の値を保存
    evaluation_values = []

    x = np.zeros((N,d)) #蜂の配置
    for i in range(N):
        x[i] = [np.random.uniform(Kp_min, Kp_max),
                np.random.uniform(Ki_min, Ki_max),
                np.random.uniform(Kd_min, Kd_max)]
    '''
    x = np.zeros((N,d)) #蜂の配置
    for i in range(N):
        x[i] = (xmax-xmin)*np.random.rand(d) + xmin
    '''    
    # 繰り返し
    best_value = []
    x_best_value = [0, 0, 0]
    z = np.zeros(N)
    best = float('-inf')  # 初期値をマイナス無限大に設定
    for t in range(G):
        # 働き蜂employee bee step
        for i in range(N):
            v = x.copy()
            k = i
            while k == i:
                k = np.random.randint(N)
            j = np.random.randint(d)
            r = np.random.rand() # 0から1までの一様乱数
            v[i,j] = x[i,j] + r * (x[i,j] - x[k,j])

            # 制約範囲内でのみ探索を許可
            v[i, j] = max(min(v[i, j], Kp_max), Kp_min) if j == 0 else v[i, j]
            v[i, j] = max(min(v[i, j], Ki_max), Ki_min) if j == 1 else v[i, j]
            v[i, j] = max(min(v[i, j], Kd_max), Kd_min) if j == 2 else v[i, j]


            if fitness_function(*x[i]) < fitness_function(*v[i]): #適合度に基づいて働き蜂の情報を更新
                x[i] = v[i]
            if fitness_function(*x[i]) <= fitness_function(*v[i]):
                s[i] = 0
            else: s[i] += 1

        # 傍観蜂onlooker bee step
        w = []
        for i in range(N):
            w.append(fitness_function(*x[i]))

        for i in range(N):
            i = roulette_choice(w)
            j = np.random.randint(d)
            r = np.random.rand() # 0から1までの一様乱数
            #v[i,j] = x[i,j] + r * (x[i,j] - x[k,j])
            # パラメータを変更し、制約範囲内に収める
            v[i][j] = max(min(x[i][j] + r * (x[i,j] - x[k,j]), Kp_max if j == 0 else Ki_max if j == 1 else Kd_max), Kp_min if j == 0 else Ki_min if j == 1 else Kd_min)

            if fitness_function(*x[i]) < fitness_function(*v[i]):
                x[i] = v[i]
            if fitness_function(*x[i]) <= fitness_function(*v[i]):
                s[i] = 0
            else: s[i] += 1

        # 斥候蜂scout bee step
        for i in range(N):
            if s[i] >= lim:
                for j in range(d):
                    #x[i,j] = (xmax-xmin)*np.random.rand()+ xmin
                    # パラメータをランダムに初期化し、制約範囲内に収める
                    x[i][j] = np.random.uniform(Kp_min if j == 0 else Ki_min if j == 1 else Kd_min,
                                       Kp_max if j == 0 else Ki_max if j == 1 else Kd_max)
                s[i] = 0

        # 最良個体の発見
        for i in range(N):
            z[i] = fitness_function(*x[i])
            if z[i] > best:
                best = z[i]
                x_best = x[i].copy()

        #best_value.append(objective_function(*x_best))
        #x_best_value.append(x_best)
        evaluation_values.append(fitness_function(*x_best))
        
    # 繰り返し回数に対する評価関数の値をプロット
    plt.plot(range(len(evaluation_values)), evaluation_values)
    plt.xlabel('Iteration')
    plt.ylabel('Evaluation Value')
    plt.grid(True)

    # タイトルをサブプロットとして設定
    plt.suptitle('Evaluation Value', fontsize=16, y=0.06)
    
    # グラフの配置を調整
    plt.subplots_adjust(bottom=0.2)
    
    plt.show()

    return x_best


x_best = ABC_algo()
convergence_value = objective_function(*x_best)

# 最も優れたPIDパラメータとその評価値を出力
print("最適なPIDパラメータ:", x_best)
print("収束値:", convergence_value[-1])
print("評価値:", fitness_function(*x_best))
print("収束誤差:", error)
print("収束時間:", settling_time)
print("オーバーシュート:", overshoot)

# シミュレーション用時間設定
time = np.linspace(0, 10, 1000)

# 制御対象の伝達関数を定義 (例: 2次遅れ系)
num = [2]
den = [1, 0.6, 1]

# 制御対象の伝達関数をシステムとして定義
system = ctrl.TransferFunction(num, den)

# PID制御器を作成
controller = ctrl.TransferFunction([x_best[2], x_best[0], x_best[1]], [1, 0])

# システムと制御器を接続
closed_loop_system = ctrl.feedback(controller * system)

feedback_connection = ctrl.TransferFunction([2*x_best[2], 2*x_best[0], 2*x_best[1]], [1, 2*x_best[2]+0.6, 2*x_best[0]+1, 2*x_best[1]])

# ステップ応答を計算
t, y = ctrl.step_response(feedback_connection, time)

# 結果をプロット
plt.plot(t, y)
plt.xlabel('Time [s]')
plt.ylabel('Output')
plt.grid(True)

# タイトルをサブプロットとして設定
plt.suptitle('Step Response', fontsize=16, y=0.06)

# グラフの配置を調整
plt.subplots_adjust(bottom=0.2)

plt.show()
