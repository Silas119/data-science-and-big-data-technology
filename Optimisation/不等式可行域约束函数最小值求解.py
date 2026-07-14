import numpy as np

# 步骤1：根据等式约束，用y表示x和z
def get_x(y):
    return 2 - y  # x + y = 2 → x = 2 - y

def get_z(y):
    return 3 - y  # y + z = 3 → z = 3 - y

# 步骤2：确定y的可行域（解不等式3y² - 10y + 3 ≤ 0）
# 解方程3y² - 10y + 3 = 0
a, b, c = 3, -10, 3
discriminant = b**2 - 4*a*c
y1 = (10 - np.sqrt(discriminant)) / (2*a)  # 较小根
y2 = (10 + np.sqrt(discriminant)) / (2*a)  # 较大根
print(f"y的可行域：[{y1:.4f}, {y2:.4f}]")  # 输出：[0.3333, 3.0000]

# 步骤3：目标函数（关于y的一元函数）
def f(y):
    x = get_x(y)
    z = get_z(y)
    return np.sin(x) + np.cos(y) + z

# 步骤4：由于函数单调递减，最小值在y的最大值处（y=3）
y_opt = y2  # 可行域右端点
x_opt = get_x(y_opt)
z_opt = get_z(y_opt)
f_opt = f(y_opt)

# 验证约束条件
constraint1 = x_opt**2 + y_opt**2 + z_opt**2  # 应≤10
constraint2 = x_opt + y_opt  # 应=2
constraint3 = y_opt + z_opt  # 应=3

print("\n解析法最优解：")
print(f"x = {x_opt:.4f}, y = {y_opt:.4f}, z = {z_opt:.4f}")
print(f"目标函数值：{f_opt:.4f}")
print(f"约束验证：x²+y²+z² = {constraint1:.4f}，x+y = {constraint2:.4f}，y+z = {constraint3:.4f}")
