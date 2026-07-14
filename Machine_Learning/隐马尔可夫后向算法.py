import numpy as np

def backward_hmm(pi: np.ndarray, A: np.ndarray, B: np.ndarray, O: list | np.ndarray) -> tuple[float, np.ndarray]:
    """
    隐马尔可夫模型后向算法，计算观测序列概率 P(O|λ)

    参数:
        pi: np.ndarray, 初始状态概率向量, shape=(N,)
        A: np.ndarray, 状态转移矩阵, shape=(N,N), A[i][j] = P(下一个状态j | 当前状态i)
        B: np.ndarray, 观测概率矩阵, shape=(N,M), B[i][o] = P(观测o | 状态i)
        O: list | np.ndarray, 观测序列, 元素为观测的0-based索引, shape=(T,)

    返回:
        P: float, 观测序列的概率 P(O|λ)
        beta: np.ndarray, 后向概率矩阵, shape=(T,N), beta[t][i] 表示t时刻状态i的后向概率
    """
    N = pi.shape[0]  # 隐藏状态数量
    T = len(O)  # 观测序列长度

    # 初始化后向概率矩阵 (T行N列)
    beta = np.zeros((T, N), dtype=np.float64)

    # 步骤1: 初值 (t=T-1对应最后一个观测o_T)
    beta[T - 1] = np.ones(N, dtype=np.float64)  # β_T(i) = 1

    # 步骤2: 递推 (t从T-2到0，对应原算法t=T-1到1)
    for t in range(T - 2, -1, -1):
        # 向量化计算 sum_{j=0}^{N-1} A[i,j] * B[j, O[t+1]] * beta[t+1,j]
        beta[t] = A @ (B[:, O[t + 1]] * beta[t + 1])

    # 步骤3: 终止，计算 P(O|λ) = sum_i π_i * b_i(o_1) * β_1(i) (对应t=0)
    P = np.sum(pi * B[:, O[0]] * beta[0])

    return P, beta
