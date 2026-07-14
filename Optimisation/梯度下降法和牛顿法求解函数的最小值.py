import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams

rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
rcParams['axes.unicode_minus'] = False


# 定义目标函数 f(x1, x2) = 2x1^2 + 4x2^2 - 4x1*x2 + 5x1 - 2x2 + 7
def f(x):
    x1, x2 = x[0], x[1]
    return 2 * x1 ** 2 + 4 * x2 ** 2 - 4 * x1 * x2 + 5 * x1 - 2 * x2 + 7


# 计算梯度 ∇f = [∂f/∂x1, ∂f/∂x2]
def gradient(x):
    x1, x2 = x[0], x[1]
    df_dx1 = 4 * x1 - 4 * x2 + 5
    df_dx2 = 8 * x2 - 4 * x1 - 2
    return np.array([df_dx1, df_dx2])


# 计算Hessian矩阵（对二次函数为常数）
def hessian(x):
    return np.array([[4, -4],
                     [-4, 8]])


# 梯度下降法
def gradient_descent(x0, tol=1e-5, max_iter=10000, alpha=0.1):
    x = x0.copy()
    history = [x.copy()]
    f_values = [f(x)]
    grad_norms = [np.linalg.norm(gradient(x))]

    for i in range(max_iter):
        grad = gradient(x)
        grad_norm = np.linalg.norm(grad)

        if grad_norm <= tol:
            break

        x = x - alpha * grad
        history.append(x.copy())
        f_values.append(f(x))
        grad_norms.append(grad_norm)

    return {
        'x': x,
        'f': f(x),
        'iterations': i + 1,
        'history': np.array(history),
        'f_values': f_values,
        'grad_norms': grad_norms
    }


# 牛顿法
def newton_method(x0, tol=1e-5, max_iter=10000):
    x = x0.copy()
    history = [x.copy()]
    f_values = [f(x)]
    grad_norms = [np.linalg.norm(gradient(x))]

    for i in range(max_iter):
        grad = gradient(x)
        grad_norm = np.linalg.norm(grad)

        if grad_norm <= tol:
            break

        H = hessian(x)
        try:
            d = np.linalg.solve(H, -grad)
            x = x + d
        except np.linalg.LinAlgError:
            print("Hessian矩阵奇异，无法求解")
            break

        history.append(x.copy())
        f_values.append(f(x))
        grad_norms.append(grad_norm)

    return {
        'x': x,
        'f': f(x),
        'iterations': i + 1,
        'history': np.array(history),
        'f_values': f_values,
        'grad_norms': grad_norms
    }


# 初始点
x0 = np.array([1.0, 2.0])
tol = 1e-5

# 运行两种算法
print("=" * 60)
print("优化问题: f(x1, x2) = 2x1² + 4x2² - 4x1x2 + 5x1 - 2x2 + 7")
print("=" * 60)

# 梯度下降法
print("\n【梯度下降法】")
result_gd = gradient_descent(x0, tol=tol, alpha=0.1)
print(f"初始点: x⁽⁰⁾ = {x0}")
print(f"最优解: x* = {result_gd['x']}")
print(f"最优值: f(x*) = {result_gd['f']:.10f}")
print(f"迭代次数: {result_gd['iterations']}")
print(f"最终梯度范数: ||∇f(x*)|| = {result_gd['grad_norms'][-1]:.2e}")

# 牛顿法
print("\n【牛顿法】")
result_newton = newton_method(x0, tol=tol)
print(f"初始点: x⁽⁰⁾ = {x0}")
print(f"最优解: x* = {result_newton['x']}")
print(f"最优值: f(x*) = {result_newton['f']:.10f}")
print(f"迭代次数: {result_newton['iterations']}")
print(f"最终梯度范数: ||∇f(x*)|| = {result_newton['grad_norms'][-1]:.2e}")

# 比较分析
print("\n" + "=" * 60)
print("【算法比较】")
print("=" * 60)
print(f"迭代次数比: 梯度下降({result_gd['iterations']}) vs 牛顿法({result_newton['iterations']})")
speedup = result_gd['iterations'] / max(result_newton['iterations'], 1)
print(f"收敛速度: 牛顿法快 {speedup:.2f} 倍")
print(f"解的精度差异: {abs(result_gd['f'] - result_newton['f']):.2e}")

# 可视化
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 1. 迭代路径图
ax1 = axes[0, 0]
x1_range = np.linspace(-2, 2, 100)
x2_range = np.linspace(-1, 3, 100)
X1, X2 = np.meshgrid(x1_range, x2_range)
# 向量化计算目标函数，避免双重循环
Z = 2 * X1 ** 2 + 4 * X2 ** 2 - 4 * X1 * X2 + 5 * X1 - 2 * X2 + 7

contour = ax1.contour(X1, X2, Z, levels=20, alpha=0.6)
ax1.clabel(contour, inline=True, fontsize=8)

gd_path = result_gd['history']
newton_path = result_newton['history']

ax1.plot(gd_path[:, 0], gd_path[:, 1], 'b.-', label='梯度下降法', linewidth=2, markersize=5)
ax1.plot(newton_path[:, 0], newton_path[:, 1], 'r.-', label='牛顿法', linewidth=2, markersize=5)
ax1.plot(x0[0], x0[1], 'go', markersize=10, label='初始点')
ax1.plot(result_gd['x'][0], result_gd['x'][1], 'b*', markersize=15, label='梯度下降终点')
ax1.plot(result_newton['x'][0], result_newton['x'][1], 'r*', markersize=15, label='牛顿法终点')

ax1.set_xlabel('x₁')
ax1.set_ylabel('x₂')
ax1.set_title('迭代路径对比')
ax1.legend()
ax1.grid(True, alpha=0.3)

# 2. 函数值下降曲线
ax2 = axes[0, 1]
ax2.semilogy(range(len(result_gd['f_values'])), result_gd['f_values'], 'b.-', label='梯度下降法')
ax2.semilogy(range(len(result_newton['f_values'])), result_newton['f_values'], 'r.-', label='牛顿法')
ax2.set_xlabel('迭代次数')
ax2.set_ylabel('函数值 f(x) (对数尺度)')
ax2.set_title('函数值收敛曲线')
ax2.legend()
ax2.grid(True, alpha=0.3)

# 3. 梯度范数下降曲线
ax3 = axes[1, 0]
ax3.semilogy(range(len(result_gd['grad_norms'])), result_gd['grad_norms'], 'b.-', label='梯度下降法')
ax3.semilogy(range(len(result_newton['grad_norms'])), result_newton['grad_norms'], 'r.-', label='牛顿法')
ax3.axhline(y=tol, color='k', linestyle='--', alpha=0.5, label=f'终止条件({tol})')
ax3.set_xlabel('迭代次数')
ax3.set_ylabel('梯度范数 ||∇f(x)|| (对数尺度)')
ax3.set_title('梯度范数收敛曲线')
ax3.legend()
ax3.grid(True, alpha=0.3)

# 4. 前10次迭代的梯度范数柱状图
ax4 = axes[1, 1]
min_len = min(len(result_gd['grad_norms']), len(result_newton['grad_norms']))
x_pos = np.arange(min(10, min_len))
bar_width = 0.35

if min_len >= 10:
    gd_bars = [result_gd['grad_norms'][i] for i in x_pos]
    newton_bars = [result_newton['grad_norms'][i] for i in x_pos]

    ax4.bar(x_pos - bar_width / 2, gd_bars, bar_width, label='梯度下降法', alpha=0.8)
    ax4.bar(x_pos + bar_width / 2, newton_bars, bar_width, label='牛顿法', alpha=0.8)
    ax4.set_xlabel('迭代次数')
    ax4.set_ylabel('梯度范数')
    ax4.set_title('前10次迭代的梯度范数对比')
    ax4.set_xticks(x_pos)
    ax4.legend()
    ax4.grid(True, alpha=0.3, axis='y')
else:
    ax4.text(0.5, 0.5, '迭代次数不足10次', ha='center', va='center', transform=ax4.transAxes)
    ax4.set_title('梯度范数对比')

plt.tight_layout()
plt.show()

print("\n可视化图表已生成！")
