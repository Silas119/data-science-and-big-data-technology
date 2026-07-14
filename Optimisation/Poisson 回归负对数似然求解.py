import numpy as np
from scipy.optimize import minimize
from scipy.special import gammaln  # 用于计算log(y!)，数值更稳定
from scipy.stats import poisson

# 设置随机种子
np.random.seed(42)


# 1. 模拟数据
# ----------------------
n = 500  # 样本量
beta0_true = 0.5  # 真实截距
beta1_true = 0.8  # 真实自变量系数

# (2) 生成自变量 X1 ~ U(0,4)
X1 = np.random.uniform(low=0, high=4, size=n)

# (3) 计算线性预测器 ηi = β0* + β1* X1i
eta = beta0_true + beta1_true * X1

# (4) 计算均值 μi = exp(ηi)
mu = np.exp(eta)

# (5) 生成响应变量 yi ~ Poisson(μi)
y = poisson.rvs(mu=mu, size=n)  # 从泊松分布抽样


# 2. 定义负对数似然函数（目标函数，需最小化）
# ----------------------
def neg_log_likelihood(beta, X, y):
    """
    计算泊松回归的负对数似然函数值
    beta: 参数向量 [β0, β1]
    X: 自变量数据
    y: 响应变量数据
    """
    beta0, beta1 = beta  # 解包参数
    eta = beta0 + beta1 * X  # 线性预测器
    mu = np.exp(eta)  # 均值（通过对数链接函数）

    # 泊松对数似然函数的总和（原问题中的L(β)）
    # log(y!) 用gammaln(y+1)计算（因log(n!)=gammaln(n+1)，数值更稳定）
    log_likelihood = np.sum(-mu + y * np.log(mu) - gammaln(y + 1))

    # 返回负对数似然（需最小化）
    return -log_likelihood


# 3. 优化求解参数估计值
# ----------------------
# 初始参数猜测（可设为0或接近真实值的任意值）
beta_init = [0.0, 0.0]

# 调用优化函数（最小化负对数似然）
# 方法选择L-BFGS-B（适用于光滑函数的无约束优化）
result = minimize(
    fun=neg_log_likelihood,
    x0=beta_init,
    args=(X1, y),  # 传入自变量和响应变量
    method='L-BFGS-B'
)

# 提取估计的参数
beta0_hat, beta1_hat = result.x

# 4. 输出结果与差距分析
# ----------------------
print("参数估计结果：")
print(f"真实截距 β0* = {beta0_true:.4f}，估计值 β0^ = {beta0_hat:.4f}")
print(f"真实系数 β1* = {beta1_true:.4f}，估计值 β1^ = {beta1_hat:.4f}")

print("\n估计值与真实值的差距：")
print(f"β0 差距：{beta0_hat - beta0_true:.4f}")
print(f"β1 差距：{beta1_hat - beta1_true:.4f}")
