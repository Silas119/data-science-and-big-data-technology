import numpy as np
import matplotlib.pyplot as plt

# 设置中文显示
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False



def pca_algorithm(X, k):
    """
    主成分分析算法 16.1

    参数：
    X: m × n 样本矩阵（每行是一个特征）
    k: 主成分个数

    返回：
    Y: k × n 样本主成分矩阵
    V: 特征向量矩阵
    explained_variance_ratio: 方差解释率
    """
    m, n = X.shape

    # 步骤1: 中心化（使每一行均值为0）
    X_mean = np.mean(X, axis=1, keepdims=True)
    X_centered = X - X_mean

    print("步骤1: 数据中心化")
    print(f"原始数据 X:\n{X}\n")
    print(f"每行均值:\n{X_mean.flatten()}\n")
    print(f"中心化后的数据 X_centered:\n{X_centered}\n")

    # 步骤2: 构造新矩阵 X' = (1/√(n-1)) * X^T
    X_prime = (1 / np.sqrt(n - 1)) * X_centered.T

    print("步骤2: 构造矩阵 X'")
    print(f"X' = (1/√{n - 1}) * X^T:")
    print(f"X' 形状: {X_prime.shape}")
    print(f"X':\n{X_prime}\n")

    # 步骤3: 对 X' 进行奇异值分解
    U, Sigma, VT = np.linalg.svd(X_prime, full_matrices=False)

    print("步骤3: 奇异值分解 (SVD)")
    print(f"U 形状: {U.shape}")
    print(f"U:\n{U}\n")
    print(f"奇异值 Σ: {Sigma}\n")
    print(f"V^T 形状: {VT.shape}")
    print(f"V^T:\n{VT}\n")

    # 步骤4: 截断，保留前k个分量
    VT_k = VT[:k, :]
    Sigma_k = Sigma[:k]

    print(f"步骤4: 保留前 k={k} 个主成分")
    print(f"前{k}个奇异值: {Sigma_k}\n")
    print(f"V^T (前{k}行):\n{VT_k}\n")

    # 步骤5: 计算样本主成分矩阵 Y = V^T * X_centered
    Y = VT_k @ X_centered

    print("步骤5: 计算主成分矩阵")
    print(f"Y = V^T * X_centered:")
    print(f"Y 形状: {Y.shape}")
    print(f"Y:\n{Y}\n")

    # 计算方差解释率
    total_variance = np.sum(Sigma ** 2)
    explained_variance = Sigma_k ** 2
    explained_variance_ratio = explained_variance / total_variance

    print("方差分析")
    print(f"总方差: {total_variance:.6f}")
    print(f"各主成分方差: {explained_variance}")
    print(f"方差解释率: {explained_variance_ratio}")
    print(f"累积方差解释率: {np.cumsum(explained_variance_ratio)}\n")

    # 验证协方差矩阵
    cov_matrix = (X_centered @ X_centered.T) / (n - 1)
    print("协方差矩阵验证")
    print(f"样本协方差矩阵:\n{cov_matrix}\n")

    return Y, VT_k, explained_variance_ratio, X_centered, X_mean


def visualize_pca(X, Y, X_centered, VT_k):
    """可视化PCA结果"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # 图1: 原始数据和主成分方向
    ax1 = axes[0]
    ax1.scatter(X_centered[0, :], X_centered[1, :], c='blue', s=100, alpha=0.6, label='样本点')
    ax1.axhline(y=0, color='k', linestyle='--', linewidth=0.5)
    ax1.axvline(x=0, color='k', linestyle='--', linewidth=0.5)

    # 绘制主成分方向
    origin = np.array([[0], [0]])
    scale = 3
    for i in range(VT_k.shape[0]):
        ax1.arrow(0, 0, VT_k[i, 0] * scale, VT_k[i, 1] * scale,
                  head_width=0.3, head_length=0.3, fc=f'C{i + 1}', ec=f'C{i + 1}',
                  linewidth=2, label=f'主成分{i + 1}')

    ax1.set_xlabel('特征1 (中心化)', fontsize=12)
    ax1.set_ylabel('特征2 (中心化)', fontsize=12)
    ax1.set_title('原始数据空间与主成分方向', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.axis('equal')

    # 图2: 主成分空间
    ax2 = axes[1]
    if Y.shape[0] == 2:
        ax2.scatter(Y[0, :], Y[1, :], c='red', s=100, alpha=0.6, label='投影后样本')
        ax2.set_ylabel('第二主成分', fontsize=12)
    else:
        ax2.scatter(Y[0, :], np.zeros_like(Y[0, :]), c='red', s=100, alpha=0.6, label='投影后样本')
        ax2.set_ylim(-1, 1)

    ax2.axhline(y=0, color='k', linestyle='--', linewidth=0.5)
    ax2.axvline(x=0, color='k', linestyle='--', linewidth=0.5)
    ax2.set_xlabel('第一主成分', fontsize=12)
    ax2.set_title('主成分空间', fontsize=14, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('pca_visualization.png', dpi=300, bbox_inches='tight')
    plt.show()


# 例题求解
print("主成分分析 - 例题 16.1")

# 输入数据
X = np.array([
    [2, 3, 3, 4, 5, 7],
    [2, 4, 5, 5, 6, 8]
])

print(f"\n原始样本矩阵 X (2×6):\n{X}\n")

# 进行PCA分析，保留2个主成分
k = 2
Y, VT_k, variance_ratio, X_centered, X_mean = pca_algorithm(X, k)

# 保留1个主成分的情况
print("\n降维到1维的情况 (k=1)")
Y_1d, VT_1, variance_ratio_1, _, _ = pca_algorithm(X, k=1)

# 可视化
visualize_pca(X, Y, X_centered, VT_k)

# 总结
print("PCA 分析总结")
print(f"1. 原始数据维度: {X.shape[0]} 维")
print(f"2. 样本数量: {X.shape[1]} 个")
print(f"3. 第一主成分方差解释率: {variance_ratio[0]:.4f} ({variance_ratio[0] * 100:.2f}%)")
if len(variance_ratio) > 1:
    print(f"4. 第二主成分方差解释率: {variance_ratio[1]:.4f} ({variance_ratio[1] * 100:.2f}%)")
print(f"5. 主成分载荷矩阵 (特征向量):\n{VT_k}")
print(f"6. 数据中心: {X_mean.flatten()}")
