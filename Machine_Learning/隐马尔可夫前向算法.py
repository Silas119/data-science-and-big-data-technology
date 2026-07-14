import numpy as np


def forward_hmm(pi: np.ndarray, A: np.ndarray, B: np.ndarray, O: list | np.ndarray) -> tuple[float, np.ndarray]:
    """
    隐马尔可夫模型前向算法，计算观测序列概率 P(O|λ)

    参数:
        pi: np.ndarray, 初始状态概率向量, shape=(N,)
        A: np.ndarray, 状态转移矩阵, shape=(N,N), A[i][j] = P(下一个状态j | 当前状态i)
        B: np.ndarray, 观测概率矩阵, shape=(N,M), B[i][o] = P(观测o | 状态i)
        O: list | np.ndarray, 观测序列, 元素为观测的0-based索引, shape=(T,)

    返回:
        P: float, 观测序列的概率 P(O|λ)
        alpha: np.ndarray, 前向概率矩阵, shape=(T,N), alpha[t][i] 表示t时刻状态i的前向概率
    """
    N = pi.shape[0]  # 隐藏状态数量
    T = len(O)  # 观测序列长度

    # 初始化前向概率矩阵 (T行N列)
    alpha = np.zeros((T, N), dtype=np.float64)

    # 步骤1: 初值 (t=0对应第一个观测o₁)
    alpha[0] = pi * B[:, O[0]]  # 向量化计算，替代循环

    # 步骤2: 递推 (t从0到T-2，对应o_{t+1}从第二个到最后一个观测)
    for t in range(T - 1):
        # 向量化计算sum(alpha[t] * A[:,i])，再乘以B[i][O[t+1]]
        alpha[t + 1] = (alpha[t] @ A) * B[:, O[t + 1]]

    # 步骤3: 终止，求和得到最终概率
    P = np.sum(alpha[-1])

    return P, alpha
