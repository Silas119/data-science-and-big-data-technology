import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.stats import norm

# ====== Windows 系统中文字体设置 ======
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体显示中文
matplotlib.rcParams['axes.unicode_minus'] = False   # 正常显示负号（如 -0.01）
# =====================================

# 设置随机种子以便结果可重复
np.random.seed(42)

# 参数设置
n = 1000
p = 6
beta_true = np.ones(p)

# 生成数据
def generate_data(n, p, beta_true, model_type='probit'):
    """生成二元模型数据"""
    X = np.random.randn(n, p)
    linear_pred = X @ beta_true

    if model_type == 'probit':
        e = np.random.randn(n)
        Y = (linear_pred + e > 0).astype(int)
    else:  # logistic
        e = np.random.logistic(0, 1, n)
        Y = (linear_pred + e > 0).astype(int)

    return X, Y

# 链接函数
def G_probit(z):
    return norm.cdf(z)

def G_logit(z):
    return np.exp(z) / (1 + np.exp(z))

# 负对数似然函数
def neg_log_likelihood(beta, X, Y, model_type='probit'):
    linear_pred = X @ beta

    if model_type == 'probit':
        G_vals = G_probit(linear_pred)
    else:
        G_vals = G_logit(linear_pred)

    G_vals = np.clip(G_vals, 1e-10, 1 - 1e-10)
    log_likelihood = np.sum(Y * np.log(G_vals) + (1 - Y) * np.log(1 - G_vals))
    return -log_likelihood

# 梯度函数
def gradient(beta, X, Y, model_type='probit'):
    linear_pred = X @ beta

    if model_type == 'probit':
        G_vals = G_probit(linear_pred)
        g_vals = norm.pdf(linear_pred)
    else:
        G_vals = G_logit(linear_pred)
        g_vals = G_vals * (1 - G_vals)

    G_vals = np.clip(G_vals, 1e-10, 1 - 1e-10)
    grad = -np.sum(X * ((Y * g_vals / G_vals - (1 - Y) * g_vals / (1 - G_vals))[:, np.newaxis]), axis=0)
    return grad

# 参数估计函数（使用 BFGS）
def estimate_beta(X, Y, model_type='probit', initial_beta=None):
    if initial_beta is None:
        initial_beta = np.zeros(X.shape[1])

    result = minimize(
        neg_log_likelihood,
        initial_beta,
        args=(X, Y, model_type),
        method='BFGS',
        jac=gradient,
        options={'maxiter': 1000}
    )
    return result.x, result

# 主程序开始
print("=" * 60)
print("二元模型参数估计")
print("=" * 60)
print(f"\n真实参数 β = {beta_true}")
print(f"样本量 n = {n}")
print(f"维度 p = {p}\n")

# Probit 模型
print("\n" + "=" * 60)
print("Probit 模型")
print("=" * 60)
X_probit, Y_probit = generate_data(n, p, beta_true, model_type='probit')
beta_hat_probit, result_probit = estimate_beta(X_probit, Y_probit, model_type='probit')

print(f"\n估计的 β: {beta_hat_probit}")
print(f"与真实值的差异: {beta_hat_probit - beta_true}")
print(f"L2 范数: {np.linalg.norm(beta_hat_probit - beta_true):.6f}")
print(f"优化是否成功: {result_probit.success}")
print(f"迭代次数: {result_probit.nit}")

# Logit 模型
print("\n" + "=" * 60)
print("Logit 模型")
print("=" * 60)
X_logit, Y_logit = generate_data(n, p, beta_true, model_type='logistic')
beta_hat_logit, result_logit = estimate_beta(X_logit, Y_logit, model_type='logit')

print(f"\n估计的 β: {beta_hat_logit}")
print(f"与真实值的差异: {beta_hat_logit - beta_true}")
print(f"L2 范数: {np.linalg.norm(beta_hat_logit - beta_true):.6f}")
print(f"优化是否成功: {result_logit.success}")
print(f"迭代次数: {result_logit.nit}")

# 绘图
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Probit 对比
ax1 = axes[0, 0]
ax1.scatter(range(p), beta_true, label='真实值', s=100, alpha=0.7, marker='o')
ax1.scatter(range(p), beta_hat_probit, label='估计值', s=100, alpha=0.7, marker='x')
ax1.plot(range(p), beta_true, 'b--', alpha=0.3)
ax1.plot(range(p), beta_hat_probit, 'r--', alpha=0.3)
ax1.set_xlabel('参数索引')
ax1.set_ylabel('参数值')
ax1.set_title('Probit模型: 参数估计对比')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Logit 对比
ax2 = axes[0, 1]
ax2.scatter(range(p), beta_true, label='真实值', s=100, alpha=0.7, marker='o')
ax2.scatter(range(p), beta_hat_logit, label='估计值', s=100, alpha=0.7, marker='x')
ax2.plot(range(p), beta_true, 'b--', alpha=0.3)
ax2.plot(range(p), beta_hat_logit, 'r--', alpha=0.3)
ax2.set_xlabel('参数索引')
ax2.set_ylabel('参数值')
ax2.set_title('Logit模型: 参数估计对比')
ax2.legend()
ax2.grid(True, alpha=0.3)

# 误差对比
ax3 = axes[1, 0]
errors_probit = beta_hat_probit - beta_true
errors_logit = beta_hat_logit - beta_true
x_pos = np.arange(p)
width = 0.35
ax3.bar(x_pos - width/2, errors_probit, width, label='Probit', alpha=0.7)
ax3.bar(x_pos + width/2, errors_logit, width, label='Logit', alpha=0.7)
ax3.axhline(y=0, color='k', linestyle='--', linewidth=0.8)
ax3.set_xlabel('参数索引')
ax3.set_ylabel('估计误差')
ax3.set_title('估计误差对比 (估计值 - 真实值)')
ax3.set_xticks(x_pos)
ax3.legend()
ax3.grid(True, alpha=0.3, axis='y')

# 收敛过程
ax4 = axes[1, 1]
history_probit = []
history_logit = []

def callback_probit(xk):
    grad_norm = np.linalg.norm(gradient(xk, X_probit, Y_probit, 'probit'))
    history_probit.append(grad_norm)

def callback_logit(xk):
    grad_norm = np.linalg.norm(gradient(xk, X_logit, Y_logit, 'logit'))
    history_logit.append(grad_norm)

# 重新运行以记录梯度范数历史
minimize(neg_log_likelihood, np.zeros(p), args=(X_probit, Y_probit, 'probit'),
         method='BFGS', jac=gradient, callback=callback_probit, options={'maxiter': 1000})
minimize(neg_log_likelihood, np.zeros(p), args=(X_logit, Y_logit, 'logit'),
         method='BFGS', jac=gradient, callback=callback_logit, options={'maxiter': 1000})

ax4.semilogy(history_probit, label='Probit', marker='o', markersize=3)
ax4.semilogy(history_logit, label='Logit', marker='s', markersize=3)
ax4.set_xlabel('迭代步数')
ax4.set_ylabel('梯度范数 (对数尺度)')
ax4.set_title('优化收敛过程')
ax4.legend()
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('binary_model_estimation.png', dpi=300, bbox_inches='tight')
plt.show()

print("\n" + "=" * 60)
print("分析完成! 图表已生成。")
print("=" * 60)
