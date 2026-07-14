'''一、核心思路与库导入
首先明确依赖库，核心用numpy生成数据、scipy.optimize求解优化问题、sklearn实现交叉验证划分，最终通过 100 次模拟评估估计性能
'''
import numpy as np
from scipy.optimize import minimize
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt

# --------------------------
# 1. 定义全局参数（与题目一致）
# --------------------------
p = 6  # 协变量维度
n = 200  # 单次模拟样本量
sim_times = 100  # 重复模拟次数
true_beta = np.array([0.01, 1, -0.001, 0.8, -1.2, 0.6]).reshape(p, 1)  # 真实β
def create_cov_matrix(p):
    """构造协变量X的协方差矩阵Σ"""
    cov = np.zeros((p, p))
    for i in range(p):
        for j in range(p):
            cov[i, j] = 0.2 ** np.abs(i - j)  # 按题目公式计算元素
    return cov
'''
步骤 1：构造协方差矩阵与生成数据
根据题目要求，协方差矩阵Σ的元素满足σ_ij = 0.2^|i-j|，需先构造该矩阵，再生成符合正态分布的X和Y
'''
def generate_data(n, p, true_beta, cov):
    """生成单次模拟的样本(X, Y)"""
    # 生成X: n×p，服从N(0, Σ)
    X = np.random.multivariate_normal(mean=np.zeros(p), cov=cov, size=n)  # shape=(n,p)
    # 生成误差项ε: n×1，服从N(0,1)
    eps = np.random.normal(loc=0, scale=1, size=(n, 1))
    # 生成Y: Y = X^Tβ + ε（矩阵运算：X@true_beta 得到n×1）
    Y = X @ true_beta + eps
    return X, Y
'''
步骤 2：定义惩罚 GMM 的目标函数与求解
惩罚 GMM 的目标函数为||Ĝ(β)||² + λ||β||₁，其中Ĝ(β) = (1/n)X^T(Y - Xβ)。需用优化器最小化该目标函数
'''


def gmm_objective(beta, X, Y, lambda_):
    """定义惩罚GMM的目标函数：||Ĝ(β)||² + λ||β||₁"""
    n = len(Y)
    beta = beta.reshape(p, 1)  # 确保β为p×1矩阵
    # 计算经验矩Ĝ(β): p×1
    residual = Y - X @ beta  # 残差：n×1
    g_hat = (1 / n) * (X.T @ residual)  # p×1
    # 目标函数值：经验矩的二次型 + L1惩罚
    objective = np.sum(g_hat ** 2) + lambda_ * np.sum(np.abs(beta))
    return objective


def solve_penalized_gmm(X, Y, lambda_):
    """求解惩罚GMM估计：最小化目标函数"""
    # 初始值：用OLS估计作为初始值（加速收敛，OLS = (X^T X)^{-1} X^T Y）
    ols_beta = np.linalg.inv(X.T @ X) @ X.T @ Y  # p×1
    initial_beta = ols_beta.flatten()  # 优化器需1维数组

    # 调用scipy优化器：L-BFGS-B适合带约束/非光滑惩罚的优化
    result = minimize(
        fun=gmm_objective,
        x0=initial_beta,
        args=(X, Y, lambda_),
        method='L-BFGS-B',
        options={'maxiter': 1000, 'disp': False}
    )

    # 返回优化后的β估计（转为p×1矩阵）
    return result.x.reshape(p, 1)
'''
步骤 3：交叉验证选择惩罚强度 λ
通过 5 折交叉验证，从候选 λ 中选择 “验证集损失最小” 的 λ，平衡偏差与方差
'''


def select_lambda_by_cv(X, Y, lambda_candidates, k=5):
    """5折交叉验证选择最优λ"""
    kf = KFold(n_splits=k, shuffle=True, random_state=42)  # 随机划分5折
    cv_losses = np.zeros(len(lambda_candidates))  # 存储每个λ的平均验证损失

    for idx, lambda_ in enumerate(lambda_candidates):
        fold_losses = []
        for train_idx, val_idx in kf.split(X):
            # 划分训练集与验证集
            X_train, X_val = X[train_idx], X[val_idx]
            Y_train, Y_val = Y[train_idx], Y[val_idx]

            # 在训练集上求β估计
            beta_cv = solve_penalized_gmm(X_train, Y_train, lambda_)

            # 在验证集上计算损失（仅经验矩二次型，不含惩罚项）
            residual_val = Y_val - X_val @ beta_cv
            g_hat_val = (1 / len(Y_val)) * (X_val.T @ residual_val)
            val_loss = np.sum(g_hat_val ** 2)
            fold_losses.append(val_loss)

        # 计算当前λ的平均验证损失
        cv_losses[idx] = np.mean(fold_losses)

    # 选择验证损失最小的λ
    best_lambda_idx = np.argmin(cv_losses)
    best_lambda = lambda_candidates[best_lambda_idx]
    return best_lambda
'''
步骤 4：100 次重复模拟与结果评估
重复 100 次 “生成数据→选 λ→求估计” 流程，计算 100 个 β̂的均值，与真实 β 比较偏差、MSE 等指标
'''
# --------------------------
# 5. 执行100次模拟
# --------------------------
# 1. 预生成λ候选集（对数分布，覆盖0.0001到10）
lambda_candidates = np.logspace(-4, 1, 50)  # 50个候选值
# 2. 存储每次模拟的β估计
beta_estimates = np.zeros((p, sim_times))  # p×100
# 3. 构造协方差矩阵（仅需1次）
cov_matrix = create_cov_matrix(p)

# 4. 100次模拟循环
for sim in range(sim_times):
    # ① 生成样本
    X, Y = generate_data(n, p, true_beta, cov_matrix)
    # ② 交叉验证选最优λ
    best_lambda = select_lambda_by_cv(X, Y, lambda_candidates)
    # ③ 求解惩罚GMM估计
    beta_hat = solve_penalized_gmm(X, Y, best_lambda)
    # ④ 存储结果
    beta_estimates[:, sim] = beta_hat.flatten()
    # 打印进度（每10次模拟）
    if (sim + 1) % 10 == 0:
        print(f"已完成 {sim + 1}/{sim_times} 次模拟")

# --------------------------
# 6. 结果分析：与真实β比较
# --------------------------
# ① 计算100次估计的均值
beta_mean = beta_estimates.mean(axis=1).reshape(p, 1)
# ② 计算各分量的偏差（均值 - 真实值）
bias = beta_mean - true_beta
# ③ 计算各分量的均方误差（MSE = E[(β̂ - β)^2]）
mse = np.mean((beta_estimates - true_beta.flatten().reshape(p, 1)) ** 2, axis=1)

# 打印结果表格
print("\n" + "="*60)
print("100次模拟结果汇总（p=6个分量）")
print("="*60)
result_table = np.hstack([
    true_beta,  # 真实β
    beta_mean,  # 估计均值
    bias,       # 偏差
    mse.reshape(p, 1)  # MSE
])
print(f"{'分量':<6}{'真实β':<10}{'估计均值':<12}{'偏差':<10}{'MSE':<10}")
print("-"*60)
for i in range(p):
    print(f"{i+1:<6}{true_beta[i,0]:<10.4f}{beta_mean[i,0]:<12.4f}{bias[i,0]:<10.4f}{mse[i]:<10.6f}")

# --------------------------
# 7. 可视化：真实β与估计均值对比
# --------------------------
plt.rcParams['font.sans-serif'] = ['SimHei']  # 支持中文
plt.figure(figsize=(10, 6))
x = np.arange(1, p+1)  # 分量序号
width = 0.35  # 柱状图宽度

plt.bar(x - width/2, true_beta.flatten(), width, label='真实β', color='#1f77b4')
plt.bar(x + width/2, beta_mean.flatten(), width, label='100次估计均值', color='#ff7f0e')

plt.xlabel('β分量序号', fontsize=12)
plt.ylabel('β值', fontsize=12)
plt.title('带L1惩罚的经验GMM估计：真实β与100次模拟均值对比', fontsize=14)
plt.xticks(x)
plt.legend()
plt.grid(axis='y', alpha=0.3)
plt.show()
