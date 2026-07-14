import cvxpy as cp

# 定义变量
x = cp.Variable()
y = cp.Variable()
z = cp.Variable()

# 目标函数
obj = cp.Minimize(x**2 + y**2 + (z - 1)**2)

# 约束条件
constraints = [
    x + y + z == 10,
    x >= 3,
    y >= 2,
    z >= 1
]

# 优化问题
prob = cp.Problem(obj, constraints)
prob.solve()

# 输出最优解和最优值
print("Optimal value =", prob.value)
print("x =", x.value)
print("y =", y.value)
print("z =", z.value)

# 输出对偶变量（拉格朗日乘子）
print("\n对偶变量（拉格朗日乘子）：")
print("lambda (对应 x+y+z=10) =", constraints[0].dual_value)
print("mu_x (对应 x>=3) =", constraints[1].dual_value)
print("mu_y (对应 y>=2) =", constraints[2].dual_value)
print("mu_z (对应 z>=1) =", constraints[3].dual_value)

# 对偶问题最优值
# 利用强对偶性，直接输出原问题的最优值
print("\n原问题最优值 p* =", prob.value)
print("由于强对偶性成立，对偶问题最优值 d* = p* =", prob.value)
