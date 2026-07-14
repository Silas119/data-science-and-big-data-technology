import numpy as np
import cvxpy as cp

# 固定随机种子
np.random.seed(42)


def simulate_once(n=1000):
    """进行一次模拟实验，返回ATE估计值"""
    # Step 1: 生成样本数据
    X = np.random.randn(n, 4)  # 四维自变量 ~ N(0,1)
    e1, e2 = np.random.randn(n), np.random.randn(n)
    Y1 = 210 + 27.4 * X[:, 0] + 13.7 * (X[:, 1] + X[:, 2] + X[:, 3]) + e1
    Y0 = 200 - 0.5 * (27.4 * X[:, 0] + 13.7 * (X[:, 1] + X[:, 2] + X[:, 3])) + e2

    # 倾向得分 P(T=1|X)
    eta = -X[:, 0] + 0.5 * X[:, 1] - 0.25 * X[:, 2] - 0.1 * X[:, 3]
    p = 1 / (1 + np.exp(-eta))

    # 生成处理变量T
    T = np.random.binomial(1, p)
    Y = T * Y1 + (1 - T) * Y0  # 观测到的结果
    X_bar = np.mean(X, axis=0)

    # Step 2: 计算 π1（处理组权重）
    pi1 = cp.Variable(n, nonneg=True)
    constraints1 = [
        cp.sum(cp.multiply(pi1, T)) == 1,
        X.T @ (cp.multiply(pi1, T)) == X_bar
    ]
    # 修正：将 Minimize 改为 Maximize
    obj1 = cp.Maximize(cp.sum(cp.entr(pi1)))  # 最大化熵
    prob1 = cp.Problem(obj1, constraints1)
    prob1.solve(solver=cp.SCS, verbose=False, max_iters=5000)
    pi1_val = np.maximum(pi1.value, 0)  # 避免出现轻微负数

    #  Step 3: 计算 π0（对照组权重）
    pi0 = cp.Variable(n, nonneg=True)
    constraints0 = [
        cp.sum(cp.multiply(pi0, (1 - T))) == 1,
        X.T @ (cp.multiply(pi0, (1 - T))) == X_bar
    ]
    # 修正：将 Minimize 改为 Maximize
    obj0 = cp.Maximize(cp.sum(cp.entr(pi0)))  # 最大化熵
    prob0 = cp.Problem(obj0, constraints0)
    prob0.solve(solver=cp.SCS, verbose=False, max_iters=5000)
    pi0_val = np.maximum(pi0.value, 0)

    # Step 4: 计算 ATE 估计量
    Delta = np.sum(pi1_val * T * Y) - np.sum(pi0_val * (1 - T) * Y)
    return Delta


def run_simulation(reps=100, n=1000):
    """重复实验，返回平均估计值与标准差"""
    results = [simulate_once(n) for _ in range(reps)]
    return np.mean(results), np.std(results)


#  主程序入口 
if __name__ == "__main__":
    mean_delta, std_delta = run_simulation(reps=100, n=1000)
    print(f"平均估计值 Δ: {mean_delta:.3f}")
print(f"标准差: {std_delta:.3f}")
