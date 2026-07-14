import numpy as np


def viterbi_decode(pi: np.ndarray, A: np.ndarray, B: np.ndarray, O: list | np.ndarray) -> tuple[list, float]:
    """
    维特比算法解码，寻找隐马尔可夫模型的最优隐藏状态序列

    参数:
        pi: np.ndarray, 初始状态概率向量, shape=(N,)
        A: np.ndarray, 状态转移矩阵, shape=(N,N)
        B: np.ndarray, 观测概率矩阵, shape=(N,M)
        O: list | np.ndarray, 观测序列, 元素为观测的索引, shape=(T,)

    返回:
        best_path: list, 最优状态序列 (0-based 索引)
        max_prob: float, 该路径的最大概率
    """
    N = pi.shape[0]  # 隐藏状态数量
    T = len(O)  # 观测序列长度

    # 1. 初始化 delta 和 Psi 矩阵
    # delta[t][i] 表示时刻 t 处于状态 i 的最大概率
    # Psi[t][i] 表示时刻 t 处于状态 i 的最优前驱状态
    delta = np.zeros((T, N), dtype=np.float64)
    psi = np.zeros((T, N), dtype=np.int64)

    # 初始化 t=0 (对应算法 t=1)
    delta[0] = pi * B[:, O[0]]
    psi[0] = 0  # 第一个时刻无前驱，记为 0

    # 2. 递推计算 (t 从 1 到 T-1，对应算法 t=2 到 T)
    for t in range(1, T):
        for i in range(N):
            # 计算 delta[t][i] = max(delta[t-1][j] * A[j][i]) * B[i][O[t]]
            # 使用向量化运算加速 max 部分
            temp = delta[t - 1] * A[:, i]
            delta[t, i] = np.max(temp) * B[i, O[t]]
            psi[t, i] = np.argmax(temp)

    # 3. 终止：找到最终时刻的最大概率和最终状态
    P_star = np.max(delta[-1])
    i_T = np.argmax(delta[-1])

    # 4. 回溯：从后向前找回最优路径
    best_path = [0] * T
    best_path[-1] = i_T  # 设置最后一个状态

    for t in range(T - 2, -1, -1):
        best_path[t] = psi[t + 1, best_path[t + 1]]

    return best_path, P_star
