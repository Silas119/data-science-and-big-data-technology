import numpy as np
def generate_hmm_observation(A, B, pi, T):
    """
    根据隐马尔可夫模型生成观测序列
    A: 状态转移概率矩阵, shape=(N, N)
    B: 观测概率矩阵, shape=(N, M)
    pi: 初始状态分布, shape=(N,)
    T: 观测序列长度
    return: 观测序列 O (list), 状态序列 I (list)
    """
    # 获取状态数 N 和观测数 M
    N = A.shape[0]
    M = B.shape[1]

    # 存储状态序列和观测序列
    state_sequence = []
    obs_sequence = []

    # 步骤(1): 按照初始状态分布 pi 生成初始状态 i_1
    # np.random.choice 从 [0,1,...,N-1] 中按概率 pi 选择初始状态
    current_state = np.random.choice(N, p=pi)
    state_sequence.append(current_state)

    # 步骤(2): 初始化 t=1
    t = 1

    # 步骤(3)(4)(5): 循环生成后续状态和观测
    while t < T:
        # 步骤(3): 按当前状态的观测概率分布 B 生成观测 o_t
        # B[current_state] 是当前状态的观测概率分布
        current_obs = np.random.choice(M, p=B[current_state])
        obs_sequence.append(current_obs)

        # 步骤(4): 按状态转移概率分布 A 生成下一个状态 i_{t+1}
        next_state = np.random.choice(N, p=A[current_state])
        state_sequence.append(next_state)

        # 更新状态和 t
        current_state = next_state
        t += 1

    # 处理 T=1 的边界情况（无循环，仅生成初始观测）
    if T == 1:
        current_obs = np.random.choice(M, p=B[current_state])
        obs_sequence.append(current_obs)

    return obs_sequence, state_sequence


# ------------------- 示例演示 -------------------
if __name__ == "__main__":
    # 1. 定义隐马尔可夫模型参数
    # 状态数 N=2 (状态1, 状态2), 观测数 M=3 (观测0, 观测1, 观测2)
    A = np.array([
        [0.5, 0.5],  # 状态1转移到1/2的概率
        [0.3, 0.7]  # 状态2转移到1/2的概率
    ])

    B = np.array([
        [0.2, 0.4, 0.4],  # 状态1生成观测0/1/2的概率
        [0.5, 0.4, 0.1]  # 状态2生成观测0/1/2的概率
    ])

    pi = np.array([0.6, 0.4])  # 初始状态分布

    # 2. 生成观测序列，长度 T=10
    T = 10
    obs_seq, state_seq = generate_hmm_observation(A, B, pi, T)

    # 3. 打印结果
    print(f"生成的状态序列 I (长度={T}): {state_seq}")
    print(f"生成的观测序列 O (长度={T}): {obs_seq}")
