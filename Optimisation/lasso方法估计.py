from sklearn.linear_model import Lasso
import numpy as np

np.random.seed(42)

# 参数设置
n = 100   # 样本量
p = 100   # 维度

# 生成 X ~ N_p(0, I)
X = np.random.randn(n, p)

# 真实系数 β
beta_true = np.zeros(p)
beta_true[:5] = 1.0  # 前5个为1，其余为0

# 生成误差项
epsilon = np.random.randn(n)  # 标准正态分布

# 生成响应变量
y = X.dot(beta_true) + epsilon

# 创建 LASSO 模型（alpha 为正则化参数）
lasso = Lasso(alpha=0.1, max_iter=10000)

# 拟合模型
lasso.fit(X, y)

# 输出结果
print("真实系数（前10个）:", beta_true[:10])
print("LASSO估计系数（前10个）:", lasso.coef_[:10])

# 查看非零系数个数
print("非零系数个数:", np.sum(lasso.coef_ != 0))
