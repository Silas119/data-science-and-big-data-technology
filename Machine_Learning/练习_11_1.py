import numpy as np


class CRF:
    def __init__(self, num_states, num_trans_features, num_state_features, S=100):
        # 基础参数定义
        self.num_states = num_states  # 标记状态数量
        self.K1 = num_trans_features  # 转移特征总数
        self.K2 = num_state_features  # 状态特征总数
        self.K = self.K1 + self.K2  # 全部特征数量
        self.S = S  # 松弛变量常数
        self.start_state = 1  # 起始状态
        self.stop_state = 1  # 终止状态
        self.w = np.zeros(self.K)  # 模型权重初始化

    # 特征函数定义（对应教材例11.1）
    def get_feature(self, k, y_prev, y_curr, pos):
        # 转移特征 t1~t5 (k=0~4)
        if k < self.K1:
            if k == 0:
                return 1 if (y_prev == 1 and y_curr == 2 and pos in (2, 3)) else 0
            if k == 1:
                return 1 if (y_prev == 1 and y_curr == 1 and pos == 2) else 0
            if k == 2:
                return 1 if (y_prev == 2 and y_curr == 1 and pos == 3) else 0
            if k == 3:
                return 1 if (y_prev == 2 and y_curr == 1 and pos == 2) else 0
            if k == 4:
                return 1 if (y_prev == 2 and y_curr == 2 and pos == 3) else 0
        # 状态特征 s1~s4 (k=5~8)
        else:
            l = k - self.K1
            if l == 0:
                return 1 if (y_curr == 1 and pos == 1) else 0
            if l == 1:
                return 1 if (y_curr == 2 and pos in (1, 2)) else 0
            if l == 2:
                return 1 if (y_curr == 1 and pos in (2, 3)) else 0
            if l == 3:
                return 1 if (y_curr == 2 and pos == 3) else 0
        return 0

    # 计算转移函数 M_i(y_{i-1}, y_i | x)
    def get_M(self, y_prev, y_curr, pos):
        total = sum(self.w[k] * self.get_feature(k, y_prev, y_curr, pos) for k in range(self.K))
        return np.exp(total)

    # 前向算法
    def forward(self, seq_len):
        n = seq_len
        alpha = np.zeros((n + 2, self.num_states + 1))
        alpha[0][self.start_state] = 1.0
        for i in range(1, n + 2):
            for y_curr in range(1, self.num_states + 1):
                for y_prev in range(1, self.num_states + 1):
                    alpha[i][y_curr] += alpha[i - 1][y_prev] * self.get_M(y_prev, y_curr, i)
        Z = alpha[n + 1][self.stop_state]
        return alpha, Z

    # 后向算法
    def backward(self, seq_len):
        n = seq_len
        beta = np.zeros((n + 2, self.num_states + 1))
        beta[n + 1][self.stop_state] = 1.0
        for i in range(n, -1, -1):
            for y_prev in range(1, self.num_states + 1):
                for y_curr in range(1, self.num_states + 1):
                    beta[i][y_prev] += self.get_M(y_prev, y_curr, i + 1) * beta[i + 1][y_curr]
        return beta

    # 计算特征的经验期望 E_{\tilde{P}}(f_k)
    def compute_expectation_empirical(self, train_data):
        emp_exp = np.zeros(self.K)
        sample_count = len(train_data)
        for x, y in train_data:
            seq_len = len(y)
            for idx in range(seq_len):
                pos = idx + 1
                y_curr = y[idx]
                y_prev = y[idx - 1] if idx > 0 else self.start_state
                for k in range(self.K):
                    emp_exp[k] += self.get_feature(k, y_prev, y_curr, pos)
        emp_exp /= sample_count
        return emp_exp

    # 计算特征的模型期望 E_P(f_k)
    def compute_expectation_model(self, seq_len, alpha, beta, Z):
        model_exp = np.zeros(self.K)
        n = seq_len
        for k in range(self.K):
            for i in range(1, n + 2):
                for y_prev in range(1, self.num_states + 1):
                    for y_curr in range(1, self.num_states + 1):
                        feat = self.get_feature(k, y_prev, y_curr, i)
                        if feat == 0:
                            continue
                        model_exp[k] += alpha[i - 1][y_prev] * self.get_M(y_prev, y_curr, i) * beta[i][y_curr]
            model_exp[k] = model_exp[k] / Z if Z != 0 else 0
        return model_exp

    # ==================== 算法11.3 核心实现 ====================
    def algorithm_11_3(self, train_data, max_iter=100, tol=1e-5):
        """
        条件随机场改进迭代尺度算法（算法11.3）
        :param train_data: 训练数据集
        :param max_iter: 最大迭代次数
        :param tol: 收敛阈值
        :return: 训练完成的模型权重
        """
        seq_len = len(train_data[0][1])
        # 步骤1：计算经验期望
        emp_exp = self.compute_expectation_empirical(train_data)

        for iteration in range(max_iter):
            # 步骤2：前向/后向计算
            alpha, Z = self.forward(seq_len)
            beta = self.backward(seq_len)

            # 步骤3：计算模型期望
            model_exp = self.compute_expectation_model(seq_len, alpha, beta, Z)

            # 步骤4：更新权重
            w_old = self.w.copy()
            for k in range(self.K):
                emp = max(emp_exp[k], 1e-10)
                model = max(model_exp[k], 1e-10)
                delta = (1 / self.S) * np.log(emp / model)
                self.w[k] += delta

            # 步骤5：收敛判断
            delta_w = np.max(np.abs(self.w - w_old))
            if delta_w < tol:
                print(f"算法11.3 收敛，迭代次数：{iteration + 1}")
                break
        return self.w


# 测试运行
if __name__ == "__main__":
    # 模型参数配置
    num_states = 2
    num_trans_features = 5
    num_state_features = 4

    # 训练数据（教材例11.1）
    train_data = [([1, 1, 1], [1, 2, 2])]

    # 初始化模型
    crf_model = CRF(num_states, num_trans_features, num_state_features)

    # 执行算法11.3训练
    trained_weights = crf_model.algorithm_11_3(train_data)

    # 输出结果
    print("\n算法11.3 训练完成")
    print("转移特征权重：", np.round(trained_weights[:5], 4))
    print("状态特征权重：", np.round(trained_weights[5:], 4))
