# 导入优化工具库
from scipy.optimize import linprog
import numpy as np
'''
    min(-Z) -(3x1+5x2)
    s.t. 2x1+x2<=10
         x1+2x2<=12
'''
# 1. 定义目标函数系数（对应 min (-3x1 -5x2)）
c = [-3, -5]

# 2. 定义不等式约束系数（Ax ≤ b）
A = [
    [2, 1],   # 原材料约束：2x1 + x2 ≤ 10
    [1, 2]    # 工时约束：x1 + 2x2 ≤ 12
]
b = [10, 12]  # 约束右侧的资源上限

# 3. 定义变量边界（x1 ≥ 0，x2 ≥ 0）

x_bounds = [(0, None), (0, None)]

# 4. 执行线性规划求解
result = linprog(
    c=c,               # 目标函数系数
    A_ub=A,            # 不等式约束矩阵
    b_ub=b,            # 不等式约束右侧
    bounds=x_bounds,   # 变量边界
    method='highs'     # 高效求解器
)

# 5. 输出结果
print("=== 线性规划求解结果 ===")
print(f"是否求解成功：{result.success}")
print(f"最优产量：甲产品 {result.x[0]:.2f} 件，乙产品 {result.x[1]:.2f} 件")
print(f"最大总利润：{ -result.fun:.2f} 元")  # 反推回原目标函数的最大值
print(f"原材料剩余：{10 - (2*result.x[0] + result.x[1]):.2f} kg")
print(f"工时剩余：{12 - (result.x[0] + 2*result.x[1]):.2f} 小时")
