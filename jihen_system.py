import numpy as np
import matplotlib.pyplot as plt

# 時間に応じて変化する二次遅れ系の目的関数
def objective_function(t):
    omega_n = 2.0 + 1.0 * np.sin(0.05 * t)
    damping_ratio = 0.2 + 0.1 * np.sin(0.1 * t)
    return (omega_n**2, 2 * damping_ratio * omega_n)

# 時間に応じて変化する目標値
def target_value(t):
    return  3 + 0.5 * np.sin(0.05 * t)

# 時間に応じて変化する外乱
def disturbance(t):
    return 0.3 * np.sin(0.2 * t)

# シミュレーションの時間ステップ
time = np.linspace(0, 100, 1000)

# 目的関数、目標値、外乱を計算
objective_values = [objective_function(t) for t in time]
target_values = [target_value(t) for t in time]
disturbance_values = [disturbance(t) for t in time]

# プロット
plt.figure(figsize=(12, 8))
plt.plot(time, objective_values, label="目的関数")
plt.plot(time, target_values, label="目標値")
plt.plot(time, disturbance_values, label="外乱")
plt.xlabel("時間")
plt.ylabel("値")
plt.legend()
plt.grid(True)
plt.title("時間に応じて変化する関数")
plt.show()
