import numpy as np


def baum_welch_train(O, N, M, max_iter=100, tol=1e-6):
    """
    Baum-Welch 算法训练 HMM 模型参数

    参数:
        O (list/np.ndarray): 观测序列, 元素为观测的 0-based 索引, 长度为 T
        N (int): 隐藏状态数量
        M (int): 观测状态数量
        max_iter (int): 最大迭代次数
        tol (float): 收敛阈值（参数变化小于该值则停止）

    返回:
        pi (np.ndarray): 训练后的初始状态概率, shape=(N,)
        A (np.ndarray): 训练后的转移矩阵, shape=(N,N)
        B (np.ndarray): 训练后的观测矩阵, shape=(N,M)
        log_likelihoods (list): 每次迭代的对数似然值（用于观察收敛）
    """
    T = len(O)

    # ---------------- 1. 初始化模型参数 ----------------
    # 初始状态概率 pi: 随机初始化并归一化
    pi = np.random.rand(N)
    pi = pi / np.sum(pi)

    # 转移矩阵 A: 随机初始化并归一化（每行和为1）
    A = np.random.rand(N, N)
    A = A / np.sum(A, axis=1, keepdims=True)

    # 观测矩阵 B: 随机初始化并归一化
    B = np.random.rand(N, M)
    B = B / np.sum(B, axis=1, keepdims=True)

    log_likelihoods = []

    for iter in range(max_iter):
        # ---------------- 2. E 步: 计算前向/后向概率, 以及 gamma、xi ----------------
        # 前向算法
        alpha, _ = forward_scaled(pi, A, B, O)  # 使用缩放前向算法避免下溢
        # 后向算法
        beta, _ = backward_scaled(pi, A, B, O, scale)  # 使用缩放后向算法

        # 计算联合概率 P(O|λ) = sum(alpha[t] * beta[t])
        P = np.sum(alpha[0] * beta[0])
        log_likelihoods.append(np.log(P))

        # 计算 gamma_t(i) = alpha_t(i)*beta_t(i) / P
        gamma = alpha * beta / P  # shape=(T,N)

        # 计算 xi_t(i,j) = alpha_t(i)*A[i][j]*B[j][O[t+1]]*beta_{t+1}(j) / P
        xi = np.zeros((T - 1, N, N))
        for t in range(T - 1):
            denom = P  # 分母为全局概率 P
            for i in range(N):
                for j in range(N):
                    xi[t, i, j] = alpha[t, i] * A[i, j] * B[j, O[t + 1]] * beta[t + 1, j] / denom

        # ---------------- 3. M 步: 更新模型参数 ----------------
        # 3.1 更新转移矩阵 A
        A_new = np.zeros((N, N))
        for i in range(N):
            numerator = np.sum(xi[:, i, :], axis=0)  # 分子: sum(xi_t(i,j))
            denominator = np.sum(gamma[:-1, i])  # 分母: sum(gamma_t(i)) t=1~T-1
            A_new[i, :] = numerator / denominator if denominator != 0 else 1e-8

        # 3.2 更新观测矩阵 B
        B_new = np.zeros((N, M))
        for j in range(N):
            for k in range(M):
                # 分子: t=1~T 且 O[t]=k 的 gamma_t(j) 求和
                numerator = np.sum(gamma[O == k, j])
                # 分母: t=1~T 的 gamma_t(j) 求和
                denominator = np.sum(gamma[:, j])
                B_new[j, k] = numerator / denominator if denominator != 0 else 1e-8

        # 3.3 更新初始概率 pi
        pi_new = gamma[0, :].copy()  # pi = gamma_1(i)

        # ---------------- 4. 检查收敛 ----------------
        # 计算参数变化量
        delta_A = np.max(np.abs(A_new - A))
        delta_B = np.max(np.abs(B_new - B))
        delta_pi = np.max(np.abs(pi_new - pi))

        # 打印迭代信息
        if (iter + 1) % 10 == 0:
            print(f"Iter {iter + 1}, Log-Likelihood: {np.log(P):.4f}, "
                  f"Delta A: {delta_A:.6f}, Delta B: {delta_B:.6f}")

        # 收敛判断: 所有参数变化小于阈值
        if delta_A < tol and delta_B < tol and delta_pi < tol:
            print(f"收敛于第 {iter + 1} 次迭代")
            A = A_new
            B = B_new
            pi = pi_new
            break

        # 更新参数
        A = A_new
        B = B_new
        pi = pi_new

    return pi, A, B, log_likelihoods


# ---------------- 依赖的缩放前向/后向算法 ----------------
def forward_scaled(pi, A, B, O):
    T = len(O)
    N = pi.shape[0]
    alpha = np.zeros((T, N))
    scale = np.zeros(T)

    # 初值
    alpha[0] = pi * B[:, O[0]]
    scale[0] = 1.0 / np.sum(alpha[0])
    alpha[0] *= scale[0]

    # 递推
    for t in range(T - 1):
        alpha[t + 1] = (alpha[t] @ A) * B[:, O[t + 1]]
        scale[t + 1] = 1.0 / np.sum(alpha[t + 1])
        alpha[t + 1] *= scale[t + 1]

    return alpha, scale


def backward_scaled(pi, A, B, O, scale):
    T = len(O)
    N = pi.shape[0]
    beta = np.zeros((T, N))

    # 初值
    beta[T - 1] = np.ones(N) * scale[T - 1]

    # 递推
    for t in range(T - 2, -1, -1):
        beta[t] = (A @ (B[:, O[t + 1]] * beta[t + 1])) * scale[t]

    return beta, scale
