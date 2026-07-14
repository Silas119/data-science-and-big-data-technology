import numpy as np
import matplotlib.pyplot as plt
from matplotlib import font_manager
import warnings

warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


class KMeansClustering:
    """
    K均值聚类算法实现
    """

    def __init__(self, n_clusters, max_iter=100, tol=1e-4, random_state=None):
        """
        初始化
        参数:
            n_clusters: 聚类个数k
            max_iter: 最大迭代次数
            tol: 收敛阈值
            random_state: 随机种子
        """
        self.k = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        self.centers = None
        self.labels = None
        self.history = []  # 记录每次迭代的中心点

    def _euclidean_distance(self, x1, x2):
        """计算欧氏距离"""
        return np.sqrt(np.sum((x1 - x2) ** 2))

    def _initialize_centers(self, X, method='random'):
        """
        初始化聚类中心
        参数:
            X: 样本数据
            method: 初始化方法，'random'随机选择，'manual'手动指定
        """
        if self.random_state is not None:
            np.random.seed(self.random_state)

        n_samples = X.shape[0]

        if method == 'random':
            # 随机选择k个样本作为初始中心
            indices = np.random.choice(n_samples, self.k, replace=False)
            centers = X[indices].copy()
        else:
            # 手动指定（用于例14.2）
            centers = X[:self.k].copy()

        return centers

    def fit(self, X, initial_centers=None, verbose=True):
        """
        执行K均值聚类算法
        参数:
            X: 样本数据，shape=(n_samples, n_features)
            initial_centers: 初始中心点（可选）
            verbose: 是否打印详细过程
        """
        X = np.array(X)
        n_samples, n_features = X.shape

        # (1) 初始化聚类中心
        if initial_centers is not None:
            self.centers = np.array(initial_centers, dtype=float)
        else:
            self.centers = self._initialize_centers(X, method='random')

        if verbose:
            print("=" * 70)
            print("K均值聚类算法")
            print("=" * 70)
            print(f"\n样本数据 (n={n_samples}, m={n_features}):")
            for i, x in enumerate(X):
                print(f"x{i + 1} = {x}")
            print(f"\n(1) 初始化: k={self.k}个聚类中心")
            for j, center in enumerate(self.centers):
                print(f"    m{j + 1} = {center}")

        self.history.append(self.centers.copy())

        # 迭代
        for iteration in range(self.max_iter):
            if verbose:
                print(f"\n{'=' * 70}")
                print(f"第{iteration + 1}次迭代")
                print(f"{'=' * 70}")

            # (2) 对样本进行聚类划分
            clusters = [[] for _ in range(self.k)]
            labels = np.zeros(n_samples, dtype=int)

            if verbose:
                print(f"\n(2) 样本聚类划分:")

            for i, x in enumerate(X):
                # 计算样本到各中心的距离
                distances = [self._euclidean_distance(x, center)
                             for center in self.centers]

                # 分配到最近的中心
                min_idx = np.argmin(distances)
                clusters[min_idx].append(i)
                labels[i] = min_idx

                if verbose:
                    dist_str = ", ".join([f"d(x{i + 1}, m{j + 1})={d:.4f}"
                                          for j, d in enumerate(distances)])
                    print(f"    x{i + 1}: {dist_str} -> C{min_idx + 1}")

            self.labels = labels

            # 打印当前聚类结果
            if verbose:
                print(f"\n    当前聚类:")
                for j, cluster in enumerate(clusters):
                    if cluster:
                        members = ', '.join([f'x{i + 1}' for i in cluster])
                        print(f"    C{j + 1} = {{{members}}}")
                    else:
                        print(f"    C{j + 1} = {{}}")

            # (3) 计算新的聚类中心
            new_centers = np.zeros((self.k, n_features))

            if verbose:
                print(f"\n(3) 更新聚类中心:")

            for j, cluster in enumerate(clusters):
                if cluster:
                    # 计算该类所有样本的均值
                    cluster_points = X[cluster]
                    new_centers[j] = np.mean(cluster_points, axis=0)

                    if verbose:
                        points_str = ' + '.join([f'x{i + 1}' for i in cluster])
                        print(f"    m{j + 1} = ({points_str}) / {len(cluster)} = {new_centers[j]}")
                else:
                    # 如果该类为空，保持原中心
                    new_centers[j] = self.centers[j]
                    if verbose:
                        print(f"    m{j + 1} = {new_centers[j]} (类为空，保持不变)")

            # (4) 检查收敛
            center_shift = np.sum([self._euclidean_distance(new_centers[j], self.centers[j])
                                   for j in range(self.k)])

            if verbose:
                print(f"\n(4) 中心点移动距离: {center_shift:.6f}")

            self.centers = new_centers
            self.history.append(self.centers.copy())

            # 判断是否收敛
            if center_shift < self.tol:
                if verbose:
                    print(f"\n算法收敛! 总迭代次数: {iteration + 1}")
                    print("=" * 70)
                break
        else:
            if verbose:
                print(f"\n达到最大迭代次数: {self.max_iter}")
                print("=" * 70)

        # 打印最终结果
        if verbose:
            print(f"\n最终聚类结果:")
            for j in range(self.k):
                cluster_members = np.where(self.labels == j)[0]
                if len(cluster_members) > 0:
                    members = ', '.join([f'x{i + 1}' for i in cluster_members])
                    print(f"C{j + 1} = {{{members}}}")
                    print(f"    中心: m{j + 1} = {self.centers[j]}")
                else:
                    print(f"C{j + 1} = {{}}")

        return self.labels, self.centers

    def predict(self, X):
        """
        预测新样本的类别
        """
        X = np.array(X)
        if X.ndim == 1:
            X = X.reshape(1, -1)

        labels = []
        for x in X:
            distances = [self._euclidean_distance(x, center)
                         for center in self.centers]
            labels.append(np.argmin(distances))

        return np.array(labels)

    def plot_clustering(self, X, title="K均值聚类结果", show_history=True):
        """
        可视化聚类结果（仅适用于2D数据）
        """
        X = np.array(X)
        if X.shape[1] != 2:
            print("仅支持2维数据可视化")
            return

        if show_history and len(self.history) > 1:
            # 显示迭代过程
            n_iter = len(self.history)
            n_cols = min(4, n_iter)
            n_rows = (n_iter + n_cols - 1) // n_cols

            fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
            if n_iter == 1:
                axes = np.array([axes])
            axes = axes.flatten()

            colors = plt.cm.Set1(np.linspace(0, 1, self.k))

            for idx, centers in enumerate(self.history):
                ax = axes[idx]

                # 如果不是第一次迭代，计算样本分配
                if idx > 0:
                    labels = []
                    for x in X:
                        distances = [self._euclidean_distance(x, center)
                                     for center in centers]
                        labels.append(np.argmin(distances))
                    labels = np.array(labels)

                    # 绘制样本点
                    for k in range(self.k):
                        mask = labels == k
                        ax.scatter(X[mask, 0], X[mask, 1],
                                   c=[colors[k]], s=100, alpha=0.6,
                                   edgecolors='black', linewidths=1,
                                   label=f'C{k + 1}')
                else:
                    # 第一次迭代，所有点用灰色
                    ax.scatter(X[:, 0], X[:, 1], c='gray', s=100,
                               alpha=0.6, edgecolors='black', linewidths=1)

                # 绘制中心点
                ax.scatter(centers[:, 0], centers[:, 1],
                           c='red', s=300, alpha=0.8,
                           marker='*', edgecolors='black', linewidths=2,
                           label='中心点')

                # 标注样本点
                for i, x in enumerate(X):
                    ax.annotate(f'x{i + 1}', (x[0], x[1]),
                                xytext=(5, 5), textcoords='offset points',
                                fontsize=9)

                # 标注中心点
                for j, c in enumerate(centers):
                    ax.annotate(f'm{j + 1}', (c[0], c[1]),
                                xytext=(5, -15), textcoords='offset points',
                                fontsize=10, fontweight='bold', color='red')

                if idx == 0:
                    ax.set_title(f'初始化', fontsize=12, fontweight='bold')
                else:
                    ax.set_title(f'第{idx}次迭代', fontsize=12, fontweight='bold')

                ax.grid(True, alpha=0.3)
                ax.legend(loc='best', fontsize=8)

            # 隐藏多余的子图
            for idx in range(n_iter, len(axes)):
                axes[idx].axis('off')

            plt.tight_layout()
            plt.savefig('kmeans_iterations.png', dpi=150, bbox_inches='tight')
            plt.show()

        # 绘制最终结果
        fig, ax = plt.subplots(figsize=(8, 6))
        colors = plt.cm.Set1(np.linspace(0, 1, self.k))

        for k in range(self.k):
            mask = self.labels == k
            ax.scatter(X[mask, 0], X[mask, 1],
                       c=[colors[k]], s=150, alpha=0.6,
                       edgecolors='black', linewidths=1.5,
                       label=f'C{k + 1}')

        # 绘制中心点
        ax.scatter(self.centers[:, 0], self.centers[:, 1],
                   c='red', s=400, alpha=0.9,
                   marker='*', edgecolors='black', linewidths=2,
                   label='中心点', zorder=10)

        # 标注样本点
        for i, x in enumerate(X):
            ax.annotate(f'x{i + 1}', (x[0], x[1]),
                        xytext=(6, 6), textcoords='offset points',
                        fontsize=11, fontweight='bold')

        # 标注中心点
        for j, c in enumerate(self.centers):
            ax.annotate(f'm{j + 1}', (c[0], c[1]),
                        xytext=(6, -18), textcoords='offset points',
                        fontsize=12, fontweight='bold', color='red')

        ax.set_xlabel('特征1', fontsize=12, fontweight='bold')
        ax.set_ylabel('特征2', fontsize=12, fontweight='bold')
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='best', fontsize=10)

        plt.tight_layout()
        plt.savefig('kmeans_final.png', dpi=150, bbox_inches='tight')
        plt.show()


def example_14_2():
    """
    例14.2: K均值聚类算法示例
    """
    print("\n" + "=" * 70)
    print("例14.2: K均值聚类算法")
    print("=" * 70)

    # 样本数据（10个2维样本点）
    X = np.array([
        [0, 2],
        [0, 0],
        [1, 0],
        [5, 0],
        [5, 2],
        [3, 0],
        [4, 0],
        [0, 1],
        [5, 1],
        [3, 2]
    ])

    # 初始聚类中心（取前3个样本点）
    initial_centers = X[[0, 1, 2]].copy()

    print("\n问题描述:")
    print(f"给定10个样本点，用K均值聚类算法将其聚为3类 (k=3)")
    print(f"初始中心点选择前3个样本: x1, x2, x3")

    # 创建K均值聚类对象
    kmeans = KMeansClustering(n_clusters=3, max_iter=100, tol=1e-6)

    # 执行聚类
    labels, centers = kmeans.fit(X, initial_centers=initial_centers, verbose=True)

    # 可视化结果
    kmeans.plot_clustering(X, title="例14.2: K均值聚类结果 (k=3)", show_history=True)

    return kmeans, X


def additional_example():
    """
    补充示例：不同k值和初始化的比较
    """
    print("\n" + "=" * 70)
    print("补充示例: 不同参数的K均值聚类比较")
    print("=" * 70)

    # 生成模拟数据（3个明显的簇）
    np.random.seed(42)

    # 簇1: 中心在(2, 2)
    cluster1 = np.random.randn(20, 2) * 0.5 + [2, 2]

    # 簇2: 中心在(8, 3)
    cluster2 = np.random.randn(20, 2) * 0.5 + [8, 3]

    # 簇3: 中心在(5, 8)
    cluster3 = np.random.randn(20, 2) * 0.5 + [5, 8]

    X = np.vstack([cluster1, cluster2, cluster3])

    # 测试不同的k值
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    k_values = [2, 3, 4, 5, 6, 7]

    for idx, k in enumerate(k_values):
        ax = axes[idx // 3, idx % 3]

        kmeans = KMeansClustering(n_clusters=k, random_state=42)
        labels, centers = kmeans.fit(X, verbose=False)

        colors = plt.cm.Set1(np.linspace(0, 1, k))

        for j in range(k):
            mask = labels == j
            ax.scatter(X[mask, 0], X[mask, 1],
                       c=[colors[j]], s=50, alpha=0.6,
                       edgecolors='black', linewidths=0.5)

        ax.scatter(centers[:, 0], centers[:, 1],
                   c='red', s=200, alpha=0.9,
                   marker='*', edgecolors='black', linewidths=2)

        ax.set_title(f'k = {k}', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)

    plt.suptitle('不同k值的聚类结果比较', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('kmeans_different_k.png', dpi=150, bbox_inches='tight')
    plt.show()

    print("\n不同k值的聚类已完成，结果已保存为图片")


def compute_metrics(X, labels, centers):
    """
    计算聚类评价指标
    """
    # 计算簇内平方和 (Within-Cluster Sum of Squares, WCSS)
    wcss = 0
    k = len(centers)

    for j in range(k):
        cluster_points = X[labels == j]
        if len(cluster_points) > 0:
            wcss += np.sum((cluster_points - centers[j]) ** 2)

    # 计算轮廓系数等其他指标...

    return {
        'wcss': wcss,
        'n_samples': len(X),
        'n_clusters': k
    }


def elbow_method_example():
    """
    使用肘部法则确定最优k值
    """
    print("\n" + "=" * 70)
    print("肘部法则（Elbow Method）确定最优k值")
    print("=" * 70)

    # 生成数据
    np.random.seed(42)
    cluster1 = np.random.randn(30, 2) * 0.5 + [2, 2]
    cluster2 = np.random.randn(30, 2) * 0.5 + [8, 3]
    cluster3 = np.random.randn(30, 2) * 0.5 + [5, 8]
    X = np.vstack([cluster1, cluster2, cluster3])

    # 测试k从1到10
    k_range = range(1, 11)
    wcss_values = []

    for k in k_range:
        kmeans = KMeansClustering(n_clusters=k, random_state=42)
        labels, centers = kmeans.fit(X, verbose=False)
        metrics = compute_metrics(X, labels, centers)
        wcss_values.append(metrics['wcss'])
        print(f"k={k}: WCSS={metrics['wcss']:.2f}")

    # 绘制肘部图
    plt.figure(figsize=(10, 6))
    plt.plot(k_range, wcss_values, 'bo-', linewidth=2, markersize=8)
    plt.xlabel('聚类数 k', fontsize=12, fontweight='bold')
    plt.ylabel('簇内平方和 (WCSS)', fontsize=12, fontweight='bold')
    plt.title('肘部法则确定最优k值', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.xticks(k_range)

    # 标注可能的肘点
    plt.axvline(x=3, color='r', linestyle='--', alpha=0.7, label='可能的最优k=3')
    plt.legend(fontsize=10)

    plt.tight_layout()
    plt.savefig('elbow_method.png', dpi=150, bbox_inches='tight')
    plt.show()


if __name__ == "__main__":
    # 运行例14.2
    kmeans, X = example_14_2()

    # 运行补充示例
    additional_example()

    # 肘部法则示例
    elbow_method_example()
