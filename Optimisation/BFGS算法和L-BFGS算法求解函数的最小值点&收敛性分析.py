import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams

# 设置中文字体
rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
rcParams['axes.unicode_minus'] = False


# 目标函数 f(x) = (x1-2)^2 + (x1-2x2)^2 + 3
def f(x):
    x1, x2 = x[0], x[1]
    return (x1 - 2) ** 2 + (x1 - 2 * x2) ** 2 + 3


# 梯度 ∇f
def gradient(x):
    x1, x2 = x[0], x[1]
    df_dx1 = 2 * (x1 - 2) + 2 * (x1 - 2 * x2)
    df_dx2 = -4 * (x1 - 2 * x2)
    return np.array([df_dx1, df_dx2])


# Wolfe条件线搜索
def wolfe_line_search(f, grad_f, x, p, grad_x):
    """使用Wolfe条件的线搜索"""
    c1 = 1e-4
    c2 = 0.9
    alpha = 1.0
    max_iter = 50
    f_x = f(x)

    for _ in range(max_iter):
        x_new = x + alpha * p
        f_new = f(x_new)
        grad_new = grad_f(x_new)

        # Armijo条件
        if f_new <= f_x + c1 * alpha * np.dot(grad_x, p):
            # 曲率条件
            if np.dot(grad_new, p) >= c2 * np.dot(grad_x, p):
                return alpha

        alpha *= 0.5

    return alpha


# BFGS算法
def bfgs(x0, tol=1e-6, max_iter=1000):
    """BFGS拟牛顿法"""
    n = len(x0)
    x = x0.copy()
    H = np.eye(n)  # 初始Hessian逆矩阵近似为单位矩阵

    history = [x.copy()]
    f_values = [f(x)]
    grad_norms = [np.linalg.norm(gradient(x))]

    for k in range(max_iter):
        grad = gradient(x)
        grad_norm = np.linalg.norm(grad)

        if grad_norm < tol:
            break

        # 搜索方向
        p = -H @ grad

        # 线搜索
        alpha = wolfe_line_search(f, gradient, x, p, grad)

        # 更新
        s = alpha * p
        x_new = x + s
        grad_new = gradient(x_new)
        y = grad_new - grad

        # 更新H（BFGS公式）
        rho = 1.0 / (y @ s)
        if rho > 0:  # 确保正定性
            I = np.eye(n)
            H = (I - rho * np.outer(s, y)) @ H @ (I - rho * np.outer(y, s)) + rho * np.outer(s, s)

        x = x_new
        history.append(x.copy())
        f_values.append(f(x))
        grad_norms.append(np.linalg.norm(grad_new))

    return {
        'x': x,
        'f': f(x),
        'iterations': k + 1,
        'history': np.array(history),
        'f_values': f_values,
        'grad_norms': grad_norms,
        'method': 'BFGS'
    }


# L-BFGS算法
def lbfgs(x0, m=10, tol=1e-6, max_iter=1000):
    """L-BFGS算法（限制内存BFGS）"""
    x = x0.copy()

    history = [x.copy()]
    f_values = [f(x)]
    grad_norms = [np.linalg.norm(gradient(x))]

    # 存储s和y的历史
    s_history = []
    y_history = []
    rho_history = []

    for k in range(max_iter):
        grad = gradient(x)
        grad_norm = np.linalg.norm(grad)

        if grad_norm < tol:
            break

        # 计算搜索方向（L-BFGS两循环递归）
        q = grad.copy()
        alpha_list = []

        # 第一个循环（从新到旧）
        for i in range(len(s_history) - 1, -1, -1):
            alpha_i = rho_history[i] * np.dot(s_history[i], q)
            alpha_list.insert(0, alpha_i)
            q = q - alpha_i * y_history[i]

        # 初始Hessian逆矩阵近似
        if len(s_history) > 0:
            gamma = np.dot(s_history[-1], y_history[-1]) / np.dot(y_history[-1], y_history[-1])
        else:
            gamma = 1.0

        r = gamma * q

        # 第二个循环（从旧到新）
        for i in range(len(s_history)):
            beta = rho_history[i] * np.dot(y_history[i], r)
            r = r + s_history[i] * (alpha_list[i] - beta)

        p = -r

        # 线搜索
        alpha = wolfe_line_search(f, gradient, x, p, grad)

        # 更新
        s = alpha * p
        x_new = x + s
        grad_new = gradient(x_new)
        y = grad_new - grad

        # 存储s和y
        rho = 1.0 / np.dot(y, s)
        if rho > 0:  # 确保正定性
            s_history.append(s)
            y_history.append(y)
            rho_history.append(rho)

            # 只保留最近m个
            if len(s_history) > m:
                s_history.pop(0)
                y_history.pop(0)
                rho_history.pop(0)

        x = x_new
        history.append(x.copy())
        f_values.append(f(x))
        grad_norms.append(np.linalg.norm(grad_new))

    return {
        'x': x,
        'f': f(x),
        'iterations': k + 1,
        'history': np.array(history),
        'f_values': f_values,
        'grad_norms': grad_norms,
        'method': 'L-BFGS'
    }


# 主程序
print("=" * 70)
print("无约束优化问题: f(x) = (x₁-2)² + (x₁-2x₂)² + 3")
print("=" * 70)

# 初始点
x0 = np.array([0.0, 0.0])
print(f"\n初始点: x⁽⁰⁾ = {x0}")
print(f"初始函数值: f(x⁽⁰⁾) = {f(x0):.6f}")

# 理论最优解分析
print("\n【理论分析】")
print("梯度为零的条件:")
print("∂f/∂x₁ = 4x₁ - 4x₂ - 4 = 0")
print("∂f/∂x₂ = -4x₁ + 8x₂ = 0")
print("理论最优解: x* = [2, 1]ᵀ")
print(f"理论最优值: f(x*) = {f(np.array([2, 1])):.6f}")

# 运行BFGS
print("\n" + "=" * 70)
print("【BFGS算法】")
print("=" * 70)
result_bfgs = bfgs(x0)
print(f"最优解: x* = [{result_bfgs['x'][0]:.8f}, {result_bfgs['x'][1]:.8f}]")
print(f"最优值: f(x*) = {result_bfgs['f']:.10f}")
print(f"迭代次数: {result_bfgs['iterations']}")
print(f"最终梯度范数: ||∇f(x*)|| = {result_bfgs['grad_norms'][-1]:.2e}")

# 运行L-BFGS
print("\n" + "=" * 70)
print("【L-BFGS算法 (m=10)】")
print("=" * 70)
result_lbfgs = lbfgs(x0, m=10)
print(f"最优解: x* = [{result_lbfgs['x'][0]:.8f}, {result_lbfgs['x'][1]:.8f}]")
print(f"最优值: f(x*) = {result_lbfgs['f']:.10f}")
print(f"迭代次数: {result_lbfgs['iterations']}")
print(f"最终梯度范数: ||∇f(x*)|| = {result_lbfgs['grad_norms'][-1]:.2e}")

# 比较分析
print("\n" + "=" * 70)
print("【算法比较】")
print("=" * 70)
print(f"迭代次数: BFGS ({result_bfgs['iterations']}) vs L-BFGS ({result_lbfgs['iterations']})")
print(f"解的精度差异: {np.linalg.norm(result_bfgs['x'] - result_lbfgs['x']):.2e}")
print(f"函数值差异: {abs(result_bfgs['f'] - result_lbfgs['f']):.2e}")
print("\n算法特点:")
print("• BFGS: 存储完整的Hessian逆矩阵近似 (n×n矩阵)")
print("• L-BFGS: 只存储有限的历史信息 (m对向量)")
print("• L-BFGS适合大规模问题，内存效率高")

# 可视化
fig = plt.figure(figsize=(16, 10))

# 1. 迭代路径图（等高线）
ax1 = plt.subplot(2, 3, 1)
x1_range = np.linspace(-1, 3, 200)
x2_range = np.linspace(-1, 2, 200)
X1, X2 = np.meshgrid(x1_range, x2_range)
Z = np.zeros_like(X1)
for i in range(X1.shape[0]):
    for j in range(X1.shape[1]):
        Z[i, j] = f(np.array([X1[i, j], X2[i, j]]))

contour = ax1.contour(X1, X2, Z, levels=30, alpha=0.6, cmap='viridis')
ax1.clabel(contour, inline=True, fontsize=8)

bfgs_path = result_bfgs['history']
lbfgs_path = result_lbfgs['history']

ax1.plot(bfgs_path[:, 0], bfgs_path[:, 1], 'b.-', label='BFGS', linewidth=2, markersize=6)
ax1.plot(lbfgs_path[:, 0], lbfgs_path[:, 1], 'r.-', label='L-BFGS', linewidth=2, markersize=6)
ax1.plot(x0[0], x0[1], 'go', markersize=12, label='初始点', zorder=5)
ax1.plot(2, 1, 'k*', markersize=15, label='理论最优解', zorder=5)

ax1.set_xlabel('x₁', fontsize=11)
ax1.set_ylabel('x₂', fontsize=11)
ax1.set_title('迭代路径对比（等高线图）', fontsize=12, fontweight='bold')
ax1.legend(fontsize=10)
ax1.grid(True, alpha=0.3)

# 2. 3D曲面图
ax2 = plt.subplot(2, 3, 2, projection='3d')
ax2.plot_surface(X1, X2, Z, alpha=0.3, cmap='viridis')
ax2.plot(bfgs_path[:, 0], bfgs_path[:, 1],
         [f(p) for p in bfgs_path], 'b.-', linewidth=2, markersize=4, label='BFGS')
ax2.plot(lbfgs_path[:, 0], lbfgs_path[:, 1],
         [f(p) for p in lbfgs_path], 'r.-', linewidth=2, markersize=4, label='L-BFGS')
ax2.set_xlabel('x₁', fontsize=10)
ax2.set_ylabel('x₂', fontsize=10)
ax2.set_zlabel('f(x)', fontsize=10)
ax2.set_title('3D迭代路径', fontsize=12, fontweight='bold')
ax2.legend(fontsize=9)

# 3. 函数值收敛曲线
ax3 = plt.subplot(2, 3, 3)
ax3.semilogy(range(len(result_bfgs['f_values'])), result_bfgs['f_values'],
             'b.-', label='BFGS', linewidth=2, markersize=5)
ax3.semilogy(range(len(result_lbfgs['f_values'])), result_lbfgs['f_values'],
             'r.-', label='L-BFGS', linewidth=2, markersize=5)
ax3.axhline(y=3.0, color='k', linestyle='--', alpha=0.5, label='理论最优值 (3.0)')
ax3.set_xlabel('迭代次数', fontsize=11)
ax3.set_ylabel('函数值 f(x) (对数尺度)', fontsize=11)
ax3.set_title('函数值收敛曲线', fontsize=12, fontweight='bold')
ax3.legend(fontsize=10)
ax3.grid(True, alpha=0.3)

# 4. 梯度范数收敛曲线
ax4 = plt.subplot(2, 3, 4)
ax4.semilogy(range(len(result_bfgs['grad_norms'])), result_bfgs['grad_norms'],
             'b.-', label='BFGS', linewidth=2, markersize=5)
ax4.semilogy(range(len(result_lbfgs['grad_norms'])), result_lbfgs['grad_norms'],
             'r.-', label='L-BFGS', linewidth=2, markersize=5)
ax4.axhline(y=1e-6, color='k', linestyle='--', alpha=0.5, label='终止条件 (1e-6)')
ax4.set_xlabel('迭代次数', fontsize=11)
ax4.set_ylabel('梯度范数 ||∇f(x)|| (对数尺度)', fontsize=11)
ax4.set_title('梯度范数收敛曲线', fontsize=12, fontweight='bold')
ax4.legend(fontsize=10)
ax4.grid(True, alpha=0.3)

# 5. 每步下降量对比
ax5 = plt.subplot(2, 3, 5)
bfgs_decrease = [result_bfgs['f_values'][i] - result_bfgs['f_values'][i + 1]
                 for i in range(len(result_bfgs['f_values']) - 1)]
lbfgs_decrease = [result_lbfgs['f_values'][i] - result_lbfgs['f_values'][i + 1]
                  for i in range(len(result_lbfgs['f_values']) - 1)]

ax5.semilogy(range(len(bfgs_decrease)), bfgs_decrease, 'b.-',
             label='BFGS', linewidth=2, markersize=5)
ax5.semilogy(range(len(lbfgs_decrease)), lbfgs_decrease, 'r.-',
             label='L-BFGS', linewidth=2, markersize=5)
ax5.set_xlabel('迭代次数', fontsize=11)
ax5.set_ylabel('函数值下降量 (对数尺度)', fontsize=11)
ax5.set_title('每步函数值下降量', fontsize=12, fontweight='bold')
ax5.legend(fontsize=10)
ax5.grid(True, alpha=0.3)

# 6. 距离最优解的距离
ax6 = plt.subplot(2, 3, 6)
x_opt = np.array([2.0, 1.0])
bfgs_dist = [np.linalg.norm(x - x_opt) for x in result_bfgs['history']]
lbfgs_dist = [np.linalg.norm(x - x_opt) for x in result_lbfgs['history']]

ax6.semilogy(range(len(bfgs_dist)), bfgs_dist, 'b.-',
             label='BFGS', linewidth=2, markersize=5)
ax6.semilogy(range(len(lbfgs_dist)), lbfgs_dist, 'r.-',
             label='L-BFGS', linewidth=2, markersize=5)
ax6.set_xlabel('迭代次数', fontsize=11)
ax6.set_ylabel('距离最优解 ||x-x*|| (对数尺度)', fontsize=11)
ax6.set_title('收敛精度分析', fontsize=12, fontweight='bold')
ax6.legend(fontsize=10)
ax6.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print("\n✓ 可视化完成！")
