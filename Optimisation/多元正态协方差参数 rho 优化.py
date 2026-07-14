import numpy as np
from scipy.optimize import minimize_scalar
from scipy.stats import multivariate_normal

# 参数
n = 1000   # 样本量
p = 30     # 维度

# 构造协方差矩阵 Σ(ρ)
def build_Sigma(rho):
    Sigma = np.ones((p, p)) * rho
    np.fill_diagonal(Sigma, 1.0)
    return Sigma

# 生成数据
def generate_data(rho):
    Sigma = build_Sigma(rho)
    X = multivariate_normal.rvs(mean=np.zeros(p), cov=Sigma, size=n, random_state=42)
    return X

# 目标函数（示例：最小化样本协方差与目标协方差的距离）
def objective(rho):
    X = generate_data(rho)
    S = np.cov(X, rowvar=False)       # 样本协方差
    Sigma = build_Sigma(rho)
    loss = np.linalg.norm(S - Sigma, 'fro')**2
    return loss

# 在 (0,1) 范围内优化 ρ
result = minimize_scalar(objective, bounds=(0, 1), method='bounded')
rho_opt = result.x
print(f"最优 ρ = {rho_opt:.4f}, 最小目标值 = {result.fun:.4f}")

# 用最优 ρ 生成最终数据
X_final = generate_data(rho_opt)
Sigma_final = build_Sigma(rho_opt)

print("最终协方差矩阵:\n", Sigma_final)
