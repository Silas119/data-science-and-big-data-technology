import numpy as np
import pandas as pd


class NMF:
    """非负矩阵分解算法"""

    def __init__(self, n_topics=2, max_iter=100, tol=1e-4):
        """
        参数:
            n_topics: 话题个数k
            max_iter: 最大迭代次数
            tol: 收敛阈值
        """
        self.n_topics = n_topics
        self.max_iter = max_iter
        self.tol = tol
        self.W = None  # 话题矩阵
        self.H = None  # 文本表示矩阵

    def _initialize(self, m, n):
        """初始化W和H矩阵"""
        # 随机初始化W (m x k)
        self.W = np.random.rand(m, self.n_topics)
        # 对W的每一列归一化
        self.W = self.W / np.sum(self.W, axis=0, keepdims=True)

        # 随机初始化H (k x n)
        self.H = np.random.rand(self.n_topics, n)

    def _update_W(self, X):
        """
        更新W矩阵
        公式 (17.33): W_il = W_il * (XH^T)_il / (WHH^T)_il
        """
        numerator = X @ self.H.T  # m x k
        denominator = self.W @ self.H @ self.H.T  # m x k
        denominator = np.maximum(denominator, 1e-10)  # 防止除零

        self.W = self.W * (numerator / denominator)

    def _update_H(self, X):
        """
        更新H矩阵
        公式 (17.34): H_lj = H_lj * (W^TX)_lj / (W^TWH)_lj
        """
        numerator = self.W.T @ X  # k x n
        denominator = self.W.T @ self.W @ self.H  # k x n
        denominator = np.maximum(denominator, 1e-10)  # 防止除零

        self.H = self.H * (numerator / denominator)

    def _reconstruction_error(self, X):
        """计算重构误差 ||X - WH||_F"""
        return np.linalg.norm(X - self.W @ self.H, 'fro')

    def fit(self, X):
        """
        执行NMF算法
        参数:
            X: 单词-文本矩阵 (m x n)
        """
        m, n = X.shape

        # 1. 初始化
        self._initialize(m, n)

        # 记录误差
        errors = []

        # 2. 迭代
        for iter_num in range(self.max_iter):
            # (a) 更新W
            self._update_W(X)

            # (b) 更新H
            self._update_H(X)

            # 计算重构误差
            error = self._reconstruction_error(X)
            errors.append(error)

            if iter_num % 10 == 0:
                print(f"迭代 {iter_num}: 重构误差 = {error:.4f}")

            # 检查收敛
            if iter_num > 0 and abs(errors[-2] - errors[-1]) < self.tol:
                print(f"在第 {iter_num} 次迭代后收敛")
                break

        return self.W, self.H, errors


def analyze_results(W, H, words, documents):
    """分析和展示NMF结果"""
    print("\n话题矩阵 W (单词-话题分布):")
    W_df = pd.DataFrame(W,
                        index=words,
                        columns=[f'话题{i + 1}' for i in range(W.shape[1])])
    print(W_df.round(3))

    print("\n文本表示矩阵 H (话题-文本分布):")
    H_df = pd.DataFrame(H,
                        index=[f'话题{i + 1}' for i in range(H.shape[0])],
                        columns=documents)
    print(H_df.round(3))

    print("\n话题解释:")
    for topic_idx in range(W.shape[1]):
        print(f"\n话题 {topic_idx + 1} 的主要单词:")
        topic_words = W_df.iloc[:, topic_idx].sort_values(ascending=False)
        for word, score in topic_words.head(3).items():
            print(f"  {word}: {score:.3f}")

    print("\n文本的主要话题:")
    for doc_idx, doc in enumerate(documents):
        print(f"\n{doc} 的话题分布:")
        topic_dist = H_df.iloc[:, doc_idx].sort_values(ascending=False)
        for topic, score in topic_dist.items():
            print(f"  {topic}: {score:.3f}")


# 构建图17.1的单词-文本矩阵
def create_word_document_matrix():
    """创建图17.1的单词-文本矩阵"""
    words = ['airplane', 'aircraft', 'computer', 'apple', 'fruit', 'produce']
    documents = ['d1', 'd2', 'd3', 'd4']

    # 单词-文本矩阵
    X = np.array([
        [2, 0, 0, 0],  # airplane
        [0, 2, 0, 0],  # aircraft
        [0, 0, 1, 0],  # computer
        [0, 0, 2, 3],  # apple
        [0, 0, 0, 1],  # fruit
        [1, 2, 2, 1]  # produce
    ], dtype=float)

    return X, words, documents


if __name__ == "__main__":
    # 创建数据
    X, words, documents = create_word_document_matrix()

    print("原始单词-文本矩阵 X:")
    X_df = pd.DataFrame(X, index=words, columns=documents)
    print(X_df)

    # 设置随机种子以保证可重复性
    np.random.seed(42)

    # 执行NMF算法，假设有2个话题
    print("\n开始非负矩阵分解 (k=2 个话题)...")
    nmf = NMF(n_topics=2, max_iter=100)
    W, H, errors = nmf.fit(X)

    # 分析结果
    analyze_results(W, H, words, documents)

    # 验证重构
    print("\n重构矩阵 W*H:")
    X_reconstructed = W @ H
    X_recon_df = pd.DataFrame(X_reconstructed, index=words, columns=documents)
    print(X_recon_df.round(2))

    print(f"\n最终重构误差: {errors[-1]:.4f}")
