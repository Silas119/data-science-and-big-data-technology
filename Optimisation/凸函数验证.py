import numpy as np
from sympy import symbols, exp, diff, simplify
#定义一个符号变量 
# 符号计算
x = symbols('x')
g = x**2
h = exp(g)
h_second = diff(h, x, 2)#二阶导数
print("h(x) =", h)
print("h''(x) =", simplify(h_second))

# 数值计算
x_vals = np.linspace(-5, 5, 1000)# 生成 1000 个等间距的点
h_second_vals = 2 * np.exp(x_vals**2) + 4 * x_vals**2 * np.exp(x_vals**2)

# 验证凸性
is_convex = np.all(h_second_vals >= 0)
# 检查所有点的二阶导数是否都大于等于 0
print(f"\n数值验证结果: h(x) {'是' if is_convex else '不是'} 凸函数")
