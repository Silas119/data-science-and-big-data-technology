import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram

class AgglomerativeClustering:
    """
    聚合层次聚类算法实现
    使用最短距离（single linkage）作为类间距离
    """

    def __init__(self, distance_matrix):
        """
        初始化
        参数:
            distance_matrix: n×n的距离矩阵
        """
        self.D = np.array(distance_matrix, dtype=float)
        self.n = len(self.D)
        self.history = []  # 记录合并历史（用于树状图）

    def fit(self, verbose=True):
        """
        执行聚合聚类算法
        参数:
            verbose: 是否打印详细过程
        """
        # 1. 初始化：每个样本为一个类
        clusters = [{i} for i in range(self.n)]
        cluster_id = self.n  # 新类ID从n开始
        current_D = self.D.copy()
        active_idx = list(range(self.n))  # 活跃类的索引（对应距离矩阵的行/列）

        if verbose:
            print("=" * 60)
            print("聚合层次聚类算法")
            print("=" * 60)
            print(f"\n初始距离矩阵 D ({self.n}×{self.n}):")
            print(np.round(current_D, 0))
            print(f"\n(1) 初始化: 构建{self.n}个类，每个类包含一个样本")
            for i, cluster in enumerate(clusters):
                print(f"    G{i + 1} = {{{', '.join('x' + str(j + 1) for j in cluster)}}}")

        step = 2
        # 迭代合并，直到只剩1个类
        while len(active_idx) > 1:
            size = len(active_idx)
            min_dist = np.inf
            a, b = -1, -1  # 距离矩阵中最小距离的坐标

            # 寻找当前距离矩阵中最小的非零距离
            for i in range(size):
                for j in range(i + 1, size):
                    if current_D[i, j] < min_dist:
                        min_dist = current_D[i, j]
                        a, b = i, j

            # 转换为原始类的索引
            idx_i = active_idx[a]
            idx_j = active_idx[b]

            # 2. 合并两个类
            new_cluster = clusters[idx_i] | clusters[idx_j]
            clusters.append(new_cluster)
            new_id = len(clusters) - 1  # 新类的索引

            # 记录合并历史（scipy dendrogram 专用格式）
            self.history.append([idx_i, idx_j, min_dist, len(new_cluster)])

            if verbose:
                print(f"\n({step}) D{idx_i + 1}{idx_j + 1} = {min_dist:.0f} 最小")
                g_i = "{" + ", ".join(f"x{x+1}" for x in sorted(clusters[idx_i])) + "}"
                g_j = "{" + ", ".join(f"x{x+1}" for x in sorted(clusters[idx_j])) + "}"
                print(f"    合并 G{idx_i + 1} = {g_i} 和 G{idx_j + 1} = {g_j}")
                print(f"    得到新类 G{cluster_id + 1} = {{{', '.join(f'x{x+1}' for x in sorted(new_cluster))}}}")

            # 3.  核心修复：更新距离矩阵（删除旧类，添加新类）
            # 保留除被合并的两个类之外的所有活跃类
            remain_idx = [k for k in active_idx if k != idx_i and k != idx_j]
            new_size = len(remain_idx) + 1
            new_D = np.zeros((new_size, new_size))

            # 填充保留类之间的原有距离
            for i in range(len(remain_idx)):
                for j in range(len(remain_idx)):
                    new_D[i, j] = current_D[
                        active_idx.index(remain_idx[i]),
                        active_idx.index(remain_idx[j])
                    ]

            # 计算新类与保留类的最短距离
            new_distances = []
            for k in remain_idx:
                d1 = current_D[active_idx.index(idx_i), active_idx.index(k)]
                d2 = current_D[active_idx.index(idx_j), active_idx.index(k)]
                dist = min(d1, d2)
                new_distances.append(dist)

            # 填充新类的行和列
            new_D[:-1, -1] = new_distances
            new_D[-1, :-1] = new_distances
            new_D[-1, -1] = 0  # 自身距离为0

            # 更新矩阵和活跃类列表
            current_D = new_D
            active_idx = remain_idx + [new_id]

            # 打印新类与其他类的距离
            if verbose and len(active_idx) > 1:
                print(f"    新类G{cluster_id + 1}与其他类的距离:")
                for i, k in enumerate(remain_idx):
                    print(f"    D{cluster_id + 1}{k + 1} = {new_distances[i]:.0f}")

            cluster_id += 1
            step += 1

        if verbose:
            print(f"\n({step}) 所有样本聚为一类，聚类终止")
            print("=" * 60)

        return clusters, self.history

    def plot_dendrogram(self, labels=None):
        """绘制树状图"""
        if not self.history:
            print("请先运行fit()方法")
            return

        plt.figure(figsize=(10, 6))
        if labels is None:
            labels = [f'x{i + 1}' for i in range(self.n)]

        Z = np.array(self.history)
        dendrogram(Z, labels=labels)
        plt.title('聚合层次聚类树状图', fontsize=14, fontproperties='SimHei')
        plt.xlabel('样本', fontsize=12, fontproperties='SimHei')
        plt.ylabel('距离', fontsize=12, fontproperties='SimHei')
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        plt.show()

def example_14_1():
    """例14.1: 演示聚合聚类算法"""
    D = np.array([
        [0, 7, 2, 9, 3],
        [7, 0, 5, 4, 6],
        [2, 5, 0, 8, 1],
        [9, 4, 8, 0, 5],
        [3, 6, 1, 5, 0]
    ])

    print("例14.1: 聚合层次聚类")
    print("\n给定5个样本，距离矩阵为:")
    print(D)
    print()

    clustering = AgglomerativeClustering(D)
    clusters, history = clustering.fit(verbose=True)

    print("\n聚类层次结构:")
    for i in range(5, len(clusters)):
        print(f"G{i+1} = {{{', '.join(f'x{x+1}' for x in sorted(clusters[i]))}}}")

    clustering.plot_dendrogram()
    return clustering

def simulate_with_real_data():
    """使用真实坐标点演示算法"""

    print("补充示例: 使用真实坐标点")
    print("=" * 60)

    points = np.array([[0,0],[1,1],[1,0],[5,5],[5,6]])
    n = len(points)
    D = np.zeros((n,n))
    for i in range(n):
        for j in range(n):
            D[i,j] = np.sqrt(np.sum((points[i]-points[j])**2))

    print(f"\n样本点坐标:")
    for i,p in enumerate(points):
        print(f"x{i+1} = {p}")
    print(f"\n距离矩阵:")
    print(np.round(D,2))

    clustering = AgglomerativeClustering(D)
    clustering.fit(verbose=True)

    # 可视化
    fig, (ax1, ax2) = plt.subplots(1,2,figsize=(14,5))
    ax1.scatter(points[:,0], points[:,1], s=100, c='blue', alpha=0.6)
    for i,p in enumerate(points):
        ax1.annotate(f'x{i+1}', (p[0],p[1]), xytext=(5,5), textcoords='offset points')
    ax1.set_title('样本点分布', fontproperties='SimHei')
    ax1.grid(True, alpha=0.3)

    plt.sca(ax2)
    dendrogram(np.array(clustering.history), labels=[f'x{i+1}' for i in range(n)])
    ax2.set_title('聚类树状图', fontproperties='SimHei')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    clustering = example_14_1()
    simulate_with_real_data()
