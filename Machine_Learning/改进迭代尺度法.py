import numpy as np

class CRF_IIS:
    def __init__(self, num_states, num_trans_features, num_state_features, S=100):
        """
        初始化CRF的改进迭代尺度法训练器
        :param num_states: 标记状态数（例：y∈{1,2}则num_states=2）
        :param num_trans_features: 转移特征t的数量
        :param num_state_features: 状态特征s的数量
        :param S: 松弛特征的常数（足够大保证s(x,y)≥0）
        """
        self.num_states = num_states  # 标记状态数m
        self.K1 = num_trans_features  # 转移特征数K1
        self.K2 = num_state_features  # 状态特征数K2
        self.K = self.K1 + self.K2    # 总特征数K
        self.S = S                    # 松弛常数
        self.start_state = 1         # 起始状态start=1
        self.stop_state = 1          # 终止状态stop=1
        self.w = np.zeros(self.K)    #  修复：初始化权重向量w（核心修复）

    # -------------------------- 1. 特征函数定义（对应例11.1） --------------------------
    def trans_feature(self, k, y_prev, y_curr, pos):
        """转移特征t_k(y_prev, y_curr, x, pos)，pos=1/2/3对应序列位置"""
        if k == 0:  # t1: y_prev=1,y_curr=2,pos=2/3
            return 1 if (y_prev == 1 and y_curr == 2 and pos in [2,3]) else 0
        elif k == 1:  # t2: y_prev=1,y_curr=1,pos=2
            return 1 if (y_prev == 1 and y_curr == 1 and pos == 2) else 0
        elif k == 2:  # t3: y_prev=2,y_curr=1,pos=3
            return 1 if (y_prev == 2 and y_curr == 1 and pos == 3) else 0
        elif k == 3:  # t4: y_prev=2,y_curr=1,pos=2
            return 1 if (y_prev == 2 and y_curr == 1 and pos == 2) else 0
        elif k == 4:  # t5: y_prev=2,y_curr=2,pos=3
            return 1 if (y_prev == 2 and y_curr == 2 and pos == 3) else 0
        else:
            return 0

    def state_feature(self, l, y_curr, pos):
        """状态特征s_l(y_curr, x, pos)，pos=1/2/3对应序列位置"""
        if l == 0:  # s1: y=1,pos=1
            return 1 if (y_curr == 1 and pos == 1) else 0
        elif l == 1:  # s2: y=2,pos=1/2
            return 1 if (y_curr == 2 and pos in [1,2]) else 0
        elif l == 2:  # s3: y=1,pos=2/3
            return 1 if (y_curr == 1 and pos in [2,3]) else 0
        elif l == 3:  # s4: y=2,pos=3
            return 1 if (y_curr == 2 and pos == 3) else 0
        else:
            return 0

    def get_feature(self, k, y_prev, y_curr, pos):
        """统一获取特征f_k：k<K1为转移特征，≥K1为状态特征"""
        if k < self.K1:
            return self.trans_feature(k, y_prev, y_curr, pos)
        else:
            l = k - self.K1
            return self.state_feature(l, y_curr, pos)

    # -------------------------- 2. 前向-后向算法（计算α,β,Z(x)） --------------------------
    def forward(self, seq_len):
        """前向算法：seq_len=序列长度（例：3）"""
        alpha = np.zeros((seq_len + 2, self.num_states + 1))
        alpha[0][self.start_state] = 1.0  # 初始化起始状态

        for i in range(1, seq_len + 2):
            for y_curr in range(1, self.num_states + 1):
                for y_prev in range(1, self.num_states + 1):
                    feat_sum = sum(self.w[k] * self.get_feature(k, y_prev, y_curr, i) for k in range(self.K))
                    M = np.exp(feat_sum)
                    alpha[i][y_curr] += alpha[i-1][y_prev] * M
        Z = alpha[seq_len + 1][self.stop_state]
        return alpha, Z

    def backward(self, seq_len):
        """后向算法"""
        beta = np.zeros((seq_len + 2, self.num_states + 1))
        beta[seq_len + 1][self.stop_state] = 1.0  # 初始化终止状态

        for i in range(seq_len, -1, -1):
            for y_prev in range(1, self.num_states + 1):
                for y_curr in range(1, self.num_states + 1):
                    feat_sum = sum(self.w[k] * self.get_feature(k, y_prev, y_curr, i+1) for k in range(self.K))
                    M = np.exp(feat_sum)
                    beta[i][y_prev] += M * beta[i+1][y_curr]
        return beta

    # -------------------------- 3. 期望计算（修复索引越界） --------------------------
    def calc_E_tilde(self, train_data):
        """计算经验期望"""
        E_tilde = np.zeros(self.K)
        total_samples = len(train_data)

        for (x, y) in train_data:
            seq_len = len(y)
            for idx in range(seq_len):
                pos = idx + 1  # 映射为文档中的位置1/2/3
                y_curr = y[idx]
                y_prev = y[idx-1] if idx > 0 else self.start_state

                # 统计转移特征
                for k in range(self.K1):
                    E_tilde[k] += self.get_feature(k, y_prev, y_curr, pos)
                # 统计状态特征
                for k in range(self.K1, self.K):
                    E_tilde[k] += self.get_feature(k, 0, y_curr, pos)
        # 归一化
        E_tilde /= total_samples
        return E_tilde

    def calc_E_p(self, seq_len, alpha, beta, Z):
        """计算模型期望"""
        E_p = np.zeros(self.K)
        for k in range(self.K):
            for i in range(1, seq_len + 2):
                for y_prev in range(1, self.num_states + 1):
                    for y_curr in range(1, self.num_states + 1):
                        feat = self.get_feature(k, y_prev, y_curr, i)
                        if feat == 0:
                            continue
                        M_val = np.exp(sum(self.w[k_]*self.get_feature(k_, y_prev, y_curr, i) for k_ in range(self.K)))
                        E_p[k] += alpha[i-1][y_prev] * M_val * beta[i][y_curr]
            E_p[k] /= Z if Z != 0 else 1e-10
        return E_p

    # -------------------------- 4. 改进迭代尺度法训练 --------------------------
    def train(self, train_data, max_iter=100, tol=1e-5):
        seq_len = len(train_data[0][1])  # 序列长度
        E_tilde = self.calc_E_tilde(train_data)

        for iter in range(max_iter):
            alpha, Z = self.forward(seq_len)
            beta = self.backward(seq_len)
            E_p = self.calc_E_p(seq_len, alpha, beta, Z)

            # 更新权重
            w_old = self.w.copy()
            for k in range(self.K):
                E_p_k = max(E_p[k], 1e-10)
                E_tilde_k = max(E_tilde[k], 1e-10)
                delta = (1 / self.S) * np.log(E_tilde_k / E_p_k)
                self.w[k] += delta

            # 收敛判断
            delta_w = np.max(np.abs(self.w - w_old))
            if delta_w < tol:
                print(f"IIS训练收敛，迭代次数：{iter+1}")
                break
        return self.w

# -------------------------- 测试运行 --------------------------
if __name__ == "__main__":
    # 配置参数（例11.1）
    num_states = 2
    num_trans_features = 5
    num_state_features = 4
    S = 100

    # 训练数据（观测序列x，标记序列y）
    train_data = [([1,1,1], [1,2,2])]

    # 初始化并训练
    crf_iis = CRF_IIS(num_states, num_trans_features, num_state_features, S)
    optimal_w = crf_iis.train(train_data, max_iter=50, tol=1e-5)

    # 输出结果
    print("\n训练完成！最优权重w：")
    print("转移特征权重(λ1-λ5)：", np.round(optimal_w[:5], 4))
    print("状态特征权重(μ1-μ4)：", np.round(optimal_w[5:], 4))
