import numpy as np
import pandas as pd

# 定义模型参数
states = ['晴天', '阴天', '下雨']
observations = ['外出', '在家']

# 初始状态概率向量
Pi = np.array([0.5, 0.3, 0.2])

# 状态转移概率矩阵 A[i][j] 表示从状态i转移到状态j的概率
A = np.array([
    [0.8, 0.1, 0.1],  # 晴天 -> 晴天, 阴天, 下雨
    [0.3, 0.4, 0.3],  # 阴天 -> 晴天, 阴天, 下雨
    [0.2, 0.4, 0.4]   # 下雨 -> 晴天, 阴天, 下雨
])

# 观测概率矩阵 B[i][j] 表示状态i下观测到j的概率
B = np.array([
    [0.9, 0.1],  # 晴天 -> 外出, 在家
    [0.4, 0.6],  # 阴天 -> 外出, 在家
    [0.1, 0.9]   # 下雨 -> 外出, 在家
])

# 观测序列映射
obs_map = {'外出': 0, '在家': 1}


print("隐马尔可夫模型参数")

print(f"\n状态集合: {states}")
print(f"观测集合: {observations}")
print(f"\n初始状态概率向量 π:\n{Pi}")
print(f"\n状态转移概率矩阵 A:\n{pd.DataFrame(A, index=states, columns=states)}")
print(f"\n观测概率矩阵 B:\n{pd.DataFrame(B, index=states, columns=observations)}")


def forward_algorithm(obs_seq, A, B, Pi, states, obs_map, verbose=True):
    """
    前向算法
    obs_seq: 观测序列
    返回: P(O|λ)
    """
    T = len(obs_seq)  # 序列长度
    N = len(states)  # 状态数

    # 初始化前向概率矩阵
    alpha = np.zeros((T, N))

    if verbose:

        print("前向算法详细计算过程")

        print(f"\n观测序列: {obs_seq}")
        print(f"序列长度 T = {T}, 状态数 N = {N}")

    # 步骤1: 初始化 (t=1)
    obs_idx = obs_map[obs_seq[0]]
    alpha[0] = Pi * B[:, obs_idx]

    if verbose:
        print(f"\n【步骤1】初始化 (t=1, 观测={obs_seq[0]})")
        print(f"α₁(i) = π(i) × b_i({obs_seq[0]})")
        for i, state in enumerate(states):
            print(f"  α₁({state}) = {Pi[i]:.1f} × {B[i, obs_idx]:.1f} = {alpha[0, i]:.4f}")
        print(f"α₁ = {alpha[0]}")

    # 步骤2: 递推 (t=2,3,...,T)
    for t in range(1, T):
        obs_idx = obs_map[obs_seq[t]]
        if verbose:
            print(f"\n【步骤2】递推 (t={t + 1}, 观测={obs_seq[t]})")
            print(f"α_{t + 1}(j) = [Σᵢ α_{t}(i) × a_ij] × b_j({obs_seq[t]})")

        for j in range(N):
            # α_t+1(j) = [Σ_i α_t(i) * a_ij] * b_j(O_t+1)
            sum_val = np.sum(alpha[t - 1] * A[:, j])
            alpha[t, j] = sum_val * B[j, obs_idx]

            if verbose:
                print(f"\n  α_{t + 1}({states[j]}):")
                sum_detail = " + ".join([f"{alpha[t - 1, i]:.4f}×{A[i, j]:.1f}"
                                         for i in range(N)])
                print(f"    = [{sum_detail}] × {B[j, obs_idx]:.1f}")
                print(f"    = {sum_val:.4f} × {B[j, obs_idx]:.1f} = {alpha[t, j]:.6f}")

        if verbose:
            print(f"α_{t + 1} = {alpha[t]}")

    # 步骤3: 终止
    P_O = np.sum(alpha[T - 1])

    if verbose:
        print(f"\n【步骤3】终止")
        print(f"P(O|λ) = Σᵢ α_{T}(i)")
        sum_detail = " + ".join([f"{alpha[T - 1, i]:.6f}" for i in range(N)])
        print(f"       = {sum_detail}")
        print(f"       = {P_O:.8f}")

    return P_O, alpha


# 任务1
obs_seq_1 = ['外出', '在家', '外出']
P_O, alpha = forward_algorithm(obs_seq_1, A, B, Pi, states, obs_map)


print(f"【结果】P(O|λ) = {P_O:.8f}")



def viterbi_algorithm(obs_seq, A, B, Pi, states, obs_map, verbose=True):
    """
    维特比算法
    obs_seq: 观测序列
    返回: 最优状态序列
    """
    T = len(obs_seq)  # 序列长度
    N = len(states)  # 状态数

    # 初始化
    delta = np.zeros((T, N))  # 最大概率
    psi = np.zeros((T, N), dtype=int)  # 最优路径记录

    if verbose:

        print("维特比算法详细计算过程")

        print(f"\n观测序列: {obs_seq}")
        print(f"序列长度 T = {T}, 状态数 N = {N}")

    # 步骤1: 初始化 (t=1)
    obs_idx = obs_map[obs_seq[0]]
    delta[0] = Pi * B[:, obs_idx]
    psi[0] = 0

    if verbose:
        print(f"\n【步骤1】初始化 (t=1, 观测={obs_seq[0]})")
        print(f"δ₁(i) = π(i) × b_i({obs_seq[0]})")
        for i, state in enumerate(states):
            print(f"  δ₁({state}) = {Pi[i]:.1f} × {B[i, obs_idx]:.1f} = {delta[0, i]:.4f}")
        print(f"δ₁ = {delta[0]}")
        print(f"ψ₁ = {psi[0]} (初始化为0)")

    # 步骤2: 递推 (t=2,3,...,T)
    for t in range(1, T):
        obs_idx = obs_map[obs_seq[t]]
        if verbose:
            print(f"\n【步骤2】递推 (t={t + 1}, 观测={obs_seq[t]})")
            print(f"δ_{t + 1}(j) = max_i[δ_{t}(i) × a_ij] × b_j({obs_seq[t]})")
            print(f"ψ_{t + 1}(j) = argmax_i[δ_{t}(i) × a_ij]")

        for j in range(N):
            # 计算所有可能的前一状态
            prob = delta[t - 1] * A[:, j]
            delta[t, j] = np.max(prob) * B[j, obs_idx]
            psi[t, j] = np.argmax(prob)

            if verbose:
                print(f"\n  δ_{t + 1}({states[j]}):")
                prob_detail = [f"δ_{t}({states[i]})×a_{i}{j} = {delta[t - 1, i]:.4f}×{A[i, j]:.1f} = {prob[i]:.4f}"
                               for i in range(N)]
                for detail in prob_detail:
                    print(f"    {detail}")
                print(f"    max = {np.max(prob):.4f}, 来自状态 {states[psi[t, j]]}")
                print(f"    δ_{t + 1}({states[j]}) = {np.max(prob):.4f} × {B[j, obs_idx]:.1f} = {delta[t, j]:.6f}")
                print(f"    ψ_{t + 1}({states[j]}) = {psi[t, j]} ({states[psi[t, j]]})")

        if verbose:
            print(f"\nδ_{t + 1} = {delta[t]}")
            print(f"ψ_{t + 1} = {psi[t]} -> [{', '.join([states[i] for i in psi[t]])}]")

    # 步骤3: 终止
    best_last_state = np.argmax(delta[T - 1])
    best_prob = delta[T - 1, best_last_state]

    if verbose:
        print(f"\n【步骤3】终止")
        print(f"P* = max_i δ_{T}(i) = {best_prob:.6f}")
        print(f"i_{T}* = argmax_i δ_{T}(i) = {best_last_state} ({states[best_last_state]})")

    # 步骤4: 回溯最优路径
    best_path = [0] * T
    best_path[T - 1] = best_last_state

    for t in range(T - 2, -1, -1):
        best_path[t] = psi[t + 1, best_path[t + 1]]

    if verbose:
        print(f"\n【步骤4】回溯最优路径")
        for t in range(T - 1, 0, -1):
            print(
                f"  i_{t}* = ψ_{t + 1}(i_{t + 1}*) = ψ_{t + 1}({states[best_path[t]]}) = {best_path[t - 1]} ({states[best_path[t - 1]]})")

    best_path_states = [states[i] for i in best_path]

    return best_path_states, best_prob, delta, psi


# 任务2
obs_seq_2 = ['外出', '在家']
best_path, best_prob, delta, psi = viterbi_algorithm(obs_seq_2, A, B, Pi, states, obs_map)


print(f"【结果】最优状态序列: {best_path}")
print(f"【结果】最优路径概率: P* = {best_prob:.6f}")
