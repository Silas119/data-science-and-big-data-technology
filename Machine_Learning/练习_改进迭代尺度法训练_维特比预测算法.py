import numpy as np


class CRF:
    def __init__(self, num_states, num_trans_features, num_state_features, S=100):
        # 基础参数
        self.num_states = num_states
        self.K1 = num_trans_features
        self.K2 = num_state_features
        self.K = self.K1 + self.K2
        self.S = S
        self.start_state = 1
        self.stop_state = 1
        self.w = np.zeros(self.K)

    # 特征函数定义
    def get_feature(self, k, y_prev, y_curr, pos):
        if k < self.K1:
            if k == 0:
                return 1 if (y_prev == 1 and y_curr == 2 and pos in [2, 3]) else 0
            if k == 1:
                return 1 if (y_prev == 1 and y_curr == 1 and pos == 2) else 0
            if k == 2:
                return 1 if (y_prev == 2 and y_curr == 1 and pos == 3) else 0
            if k == 3:
                return 1 if (y_prev == 2 and y_curr == 1 and pos == 2) else 0
            if k == 4:
                return 1 if (y_prev == 2 and y_curr == 2 and pos == 3) else 0
        else:
            l = k - self.K1
            if l == 0:
                return 1 if (y_curr == 1 and pos == 1) else 0
            if l == 1:
                return 1 if (y_curr == 2 and pos in [1, 2]) else 0
            if l == 2:
                return 1 if (y_curr == 1 and pos in [2, 3]) else 0
            if l == 3:
                return 1 if (y_curr == 2 and pos == 3) else 0
        return 0

    # 计算转移矩阵值
    def get_M(self, y_prev, y_curr, pos):
        feat_sum = sum(self.w[k] * self.get_feature(k, y_prev, y_curr, pos) for k in range(self.K))
        return np.exp(feat_sum)

    # 算法11.1 改进迭代尺度法训练
    def calc_E_tilde(self, train_data):
        E_tilde = np.zeros(self.K)
        sample_num = len(train_data)
        for x, y in train_data:
            seq_len = len(y)
            for idx in range(seq_len):
                pos = idx + 1
                y_curr = y[idx]
                y_prev = y[idx - 1] if idx > 0 else self.start_state
                for k in range(self.K):
                    E_tilde[k] += self.get_feature(k, y_prev, y_curr, pos)
        return E_tilde / sample_num

    def train(self, train_data, max_iter=50, tol=1e-5):
        seq_len = len(train_data[0][1])
        E_tilde = self.calc_E_tilde(train_data)
        for _ in range(max_iter):
            w_old = self.w.copy()
            E_p = E_tilde * 0.5
            for k in range(self.K):
                E_p_k = max(E_p[k], 1e-10)
                E_tilde_k = max(E_tilde[k], 1e-10)
                delta = (1 / self.S) * np.log(E_tilde_k / E_p_k)
                self.w[k] += delta
            if np.max(np.abs(self.w - w_old)) < tol:
                print("算法11.1训练完成")
                break
        return self.w

    # 算法11.2 维特比预测算法
    def viterbi_predict(self, seq_len):
        n = seq_len
        m = self.num_states

        # 初始化
        delta = np.zeros((n + 2, m + 1))
        psi = np.zeros((n + 2, m + 1), dtype=int)

        for j in range(1, m + 1):
            delta[1][j] = self.get_M(self.start_state, j, 1)
            psi[1][j] = self.start_state

        # 递推
        for i in range(2, n + 1):
            for j in range(1, m + 1):
                max_val = -np.inf
                max_l = self.start_state
                for l in range(1, m + 1):
                    val = delta[i - 1][l] * self.get_M(l, j, i)
                    if val > max_val:
                        max_val = val
                        max_l = l
                delta[i][j] = max_val
                psi[i][j] = max_l

        # 终止
        max_delta = -np.inf
        y_end = self.stop_state
        for j in range(1, m + 1):
            val = delta[n][j] * self.get_M(j, self.stop_state, n + 1)
            if val > max_delta:
                max_delta = val
                y_end = j

        # 回溯
        best_path = np.zeros(n + 1, dtype=int)
        best_path[n] = y_end
        for i in range(n - 1, 0, -1):
            best_path[i] = psi[i + 1][best_path[i + 1]]

        return best_path[1:]


# 测试运行
if __name__ == "__main__":
    # 参数配置
    num_states = 2
    num_trans_features = 5
    num_state_features = 4

    # 训练数据
    train_data = [([1, 1, 1], [1, 2, 2])]

    # 模型初始化
    crf = CRF(num_states, num_trans_features, num_state_features)

    # 模型训练
    crf.train(train_data)

    # 最优路径预测
    best_label = crf.viterbi_predict(seq_len=3)

    # 输出结果
    print("\n算法11.2 预测结果")
    print("观测序列 x = [1,1,1]")
    print("最优标记序列 y =", list(best_label))
