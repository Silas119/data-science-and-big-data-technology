import numpy as np
import matplotlib.pyplot as plt

# 定义函数 f(x1, x2)
def f(x):
    x1, x2 = x
    return 2*x1**2 + 4*x2**2 - 4*x1*x2 + 5*x1 - 2*x2 + 7

# 定义梯度 ∇f(x1, x2)
def gradient_f(x):
    x1, x2 = x
    return np.array([4*x1 - 4*x2 + 5, 8*x2 - 4*x1 - 2])

# 定义hessian矩阵 H(f)
def hessian_f(x):
    return np.array([[4, -4], [-4, 8]])

# 梯度下降法
def gradient_descent(x0, alpha=0.01, tol=1e-5, max_iter=1000):
    x = x0
    grad_norms = []
    for i in range(max_iter):
        grad = gradient_f(x)
        grad_norm = np.linalg.norm(grad)
        grad_norms.append(grad_norm)
        if grad_norm <= tol:
            break
        x = x - alpha * grad
    return x, grad_norms

# 牛顿法
def newton_method(x0, tol=1e-5, max_iter=1000):
    x = x0
    grad_norms = []
    for i in range(max_iter):
        grad = gradient_f(x)
        grad_norm = np.linalg.norm(grad)
        grad_norms.append(grad_norm)
        if grad_norm <= tol:
            break
        hess = hessian_f(x)
        x = x - np.linalg.inv(hess) @ grad
    return x, grad_norms

# 初始点
x0 = np.array([1, 2])

# 梯度下降法
x_gd, grad_norms_gd = gradient_descent(x0)
print("Gradient Descent Solution:", x_gd)

# 牛顿法
x_nm, grad_norms_nm = newton_method(x0)
print("Newton's Method Solution:", x_nm)

# 绘制梯度范数与迭代步数的关系图
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.plot(grad_norms_gd, label='Gradient Descent')
plt.title('Gradient Norm vs Iterations (Gradient Descent)')
plt.xlabel('Iterations')
plt.ylabel('Gradient Norm')
plt.yscale('log')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(grad_norms_nm, label="Newton's Method")
plt.title("Gradient Norm vs Iterations (Newton's Method)")
plt.xlabel('Iterations')
plt.ylabel('Gradient Norm')
plt.yscale('log')
plt.legend()

plt.tight_layout()
plt.show()
