import numpy as np
from itertools import product

# 特征函数定义
def t1(y_prev, y_curr, x, i):
    return 1 if (y_prev == 1 and y_curr == 2 and i in [2, 3]) else 0

def t2(y_prev, y_curr, x, i):
    return 1 if (y_prev == 1 and y_curr == 1 and i == 2) else 0

def t3(y_prev, y_curr, x, i):
    return 1 if (y_prev == 2 and y_curr == 1 and i == 3) else 0

def t4(y_prev, y_curr, x, i):
    return 1 if (y_prev == 2 and y_curr == 1 and i == 2) else 0

def t5(y_prev, y_curr, x, i):
    return 1 if (y_prev == 2 and y_curr == 2 and i == 3) else 0

def s1(y_curr, x, i):
    return 1 if (y_curr == 1 and i == 1) else 0

def s2(y_curr, x, i):
    return 1 if (y_curr == 2 and i in [1, 2]) else 0

def s3(y_curr, x, i):
    return 1 if (y_curr == 1 and i in [2, 3]) else 0

def s4(y_curr, x, i):
    return 1 if (y_curr == 2 and i == 3) else 0

# 权值
lambdas = [(t1, 1.0, "t1"),
           (t2, 0.6, "t2"),
           (t3, 1.0, "t3"),
           (t4, 1.0, "t4"),
           (t5, 0.2, "t5")]

mus = [(s1, 1.0, "s1"),
       (s2, 0.5, "s2"),
       (s3, 0.8, "s3"),
       (s4, 0.5, "s4")]

labels = [1, 2]
x = None
n = 3

# 工具函数：计算得分
def compute_score(y_seq, x, verbose=False):
    total_score = 0.0

    if verbose:
        print(f"标记序列 y = {y_seq}")
        print("转移特征函数贡献")

    for func, lam, name in lambdas:
        func_sum = 0
        positions = []
        for i in range(2, n + 1):
            val = func(y_seq[i-2], y_seq[i-1], x, i)
            if val != 0:
                positions.append(i)
            func_sum += val
        contribution = lam * func_sum
        total_score += contribution
        if verbose:
            pos_str = (f"在位置 {positions} 取值1") if positions else "均取值0"
            print(f"  {name}(λ={lam}): {pos_str} → Σ{name}={func_sum}, λ*Σ = {lam}×{func_sum} = {contribution:.4f}")

    if verbose:
        print("状态特征函数贡献")

    for func, mu, name in mus:
        func_sum = 0
        positions = []
        for i in range(1, n + 1):
            val = func(y_seq[i-1], x, i)
            if val != 0:
                positions.append(i)
            func_sum += val
        contribution = mu * func_sum
        total_score += contribution
        if verbose:
            pos_str = (f"在位置 {positions} 取值1") if positions else "均取值0"
            print(f"  {name}(μ={mu}): {pos_str} → Σ{name}={func_sum}, μ*Σ = {mu}×{func_sum} = {contribution:.4f}")

    return total_score

# 第1题：y=(1,2,2) 的非规范化条件概率
print("【第1题】非规范化条件概率计算：y = (1, 2, 2)")
y_target = [1, 2, 2]
score = compute_score(y_target, x, verbose=True)

print(f"\n线性组合得分（指数内）= {score:.4f}")
unnorm_prob = np.exp(score)
print(f"非规范化条件概率 P̃(y|x) = exp({score:.4f}) = {unnorm_prob:.6f}")

# 第2题：维特比算法求最优标记序列
print("\n【第2题】维特比算法求最优标记序列")

def node_score(y_curr, i, x):
    score = 0.0
    for func, mu, name in mus:
        score += mu * func(y_curr, x, i)
    return score

def edge_score(y_prev, y_curr, i, x):
    score = 0.0
    for func, lam, name in lambdas:
        score += lam * func(y_prev, y_curr, x, i)
    return score

# 预计算节点势
print("\n节点势函数（状态得分）预计算")
node_scores = {}
for i in range(1, n + 1):
    for y in labels:
        ns = node_score(y, i, x)
        node_scores[(i, y)] = ns
        print(f"  node_score(y_{i}={y}, i={i}) = {ns:.4f}")

# 预计算边势
print("\n边势函数（转移得分）预计算")
edge_scores = {}
for i in range(2, n + 1):
    for y_prev in labels:
        for y_curr in labels:
            es = edge_score(y_prev, y_curr, i, x)
            edge_scores[(i, y_prev, y_curr)] = es
            print(f"  edge_score(y_{i-1}={y_prev}→y_{i}={y_curr}, i={i}) = {es:.4f}")

# 维特比递推
print("\n维特比递推过程")

delta = [{} for _ in range(n + 1)]
psi   = [{} for _ in range(n + 1)]

print(f"\n初始化 (i=1):")
for y in labels:
    delta[1][y] = node_scores[(1, y)]
    psi[1][y]   = None
    print(f"  δ_1(y_1={y}) = node_score(y_1={y}) = {delta[1][y]:.4f}")

for i in range(2, n + 1):
    print(f"\n递推 (i={i}):")
    for y_curr in labels:
        best_score = -np.inf
        best_prev  = None
        for y_prev in labels:
            candidate = (delta[i-1][y_prev]
                        + edge_scores[(i, y_prev, y_curr)]
                        + node_scores[(i, y_curr)])
            print(f"  δ_{i-1}(y_{i-1}={y_prev}) + edge({y_prev}→{y_curr}) + node(y_{i}={y_curr})")
            print(f"    = {delta[i-1][y_prev]:.4f} + {edge_scores[(i,y_prev,y_curr)]:.4f}"
                  f" + {node_scores[(i,y_curr)]:.4f} = {candidate:.4f}")
            if candidate > best_score:
                best_score = candidate
                best_prev  = y_prev
        delta[i][y_curr] = best_score
        psi[i][y_curr]   = best_prev
        print(f"  δ_{i}(y_{i}={y_curr}) = {best_score:.4f}, ψ_{i}(y_{i}={y_curr}) = y_{i-1}={best_prev}")

print(f"\n终止 (i={n}):")
best_final_score = -np.inf
best_final_label = None
for y in labels:
    print(f"  δ_{n}(y_{n}={y}) = {delta[n][y]:.4f}")
    if delta[n][y] > best_final_score:
        best_final_score = delta[n][y]
        best_final_label = y

print(f"\n最优终止标签: y_{n}* = {best_final_label}, 最大得分 = {best_final_score:.4f}")

print(f"\n路径回溯:")
best_path = [None] * (n + 1)
best_path[n] = best_final_label
for i in range(n, 1, -1):
    best_path[i-1] = psi[i][best_path[i]]
    print(f"  ψ_{i}(y_{i}={best_path[i]}) → y_{i-1}* = {best_path[i-1]}")

optimal_y = best_path[1:]
print(f"\n最优标记序列: y* = {optimal_y}")
print(f"最优路径对数得分: {best_final_score:.4f}")
print(f"最优路径非规范化概率: exp({best_final_score:.4f}) = {np.exp(best_final_score):.6f}")

# 验证：枚举所有路径
print("\n【验证】枚举所有标记序列得分")

all_scores = {}
for y_seq in product(labels, repeat=n):
    y_list = list(y_seq)
    s = compute_score(y_list, x, verbose=False)
    all_scores[y_seq] = s

Z = sum(np.exp(s) for s in all_scores.values())

print(f"\n  {'标记序列':<18} {'线性得分':>10} {'非规范化概率':>14} {'条件概率P(y|x)':>16}")
for y_seq, s in sorted(all_scores.items(), key=lambda kv: -kv[1]):
    p_unnorm = np.exp(s)
    p_norm   = p_unnorm / Z
    marker   = " 最优" if list(y_seq) == optimal_y else ""
    y_str    = str(list(y_seq))
    print(f"  {y_str:<18} {s:>10.4f} {p_unnorm:>14.6f} {p_norm:>16.6f}{marker}")

print(f"\n规范化因子 Z(x) = {Z:.6f}")
target_key = tuple(y_target)
print(f"\ny=(1,2,2) 的非规范化概率 = {np.exp(all_scores[target_key]):.6f}")
print(f"y=(1,2,2) 的条件概率     = {np.exp(all_scores[target_key])/Z:.6f}")
