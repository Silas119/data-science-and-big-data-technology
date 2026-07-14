import sympy as sp

# 定义符号变量
x, y = sp.symbols('x y', real=True)  # 限定变量为实数域

# 定义目标函数
f = x ** 3 - 3 * x * y + y ** 3

# 1. 求一阶偏导数（梯度）
fx = sp.diff(f, x)  # 对x的偏导数：3x² - 3y
fy = sp.diff(f, y)  # 对y的偏导数：3y² - 3x

# 2. 解方程组 fx=0 且 fy=0，得到临界点（并筛选实数解）
critical_points = sp.solve((fx, fy), (x, y))

# 筛选实数临界点（排除复数解）
real_critical_points = []
for point in critical_points:
    px, py = point
    # 判断x和y是否为实数（ SymPy中用is_real属性 ）
    if px.is_real and py.is_real:
        real_critical_points.append(point)

print("实数临界点：", real_critical_points)

# 3. 求二阶偏导数，构建海塞矩阵
fxx = sp.diff(fx, x)  # 6x
fxy = sp.diff(fx, y)  # -3
fyy = sp.diff(fy, y)  # 6y

# 海塞矩阵行列式：det(H) = fxx*fyy - fxy² = 36xy - 9
det_H = fxx * fyy - fxy ** 2

# 4. 判断每个实数临界点的类型
for point in real_critical_points:
    px, py = point
    # 计算该点的二阶偏导数和行列式值
    fxx_val = fxx.subs({x: px, y: py})
    det_H_val = det_H.subs({x: px, y: py})
    # 计算函数值
    f_val = f.subs({x: px, y: py})

    # 判断类型
    if det_H_val > 0:
        if fxx_val > 0:
            print(f"点 ({px}, {py}) 是极小值点，极小值为 {f_val}")
        else:
            print(f"点 ({px}, {py}) 是极大值点，极大值为 {f_val}")
    elif det_H_val < 0:
        print(f"点 ({px}, {py}) 是鞍点，不是极值点")
    else:
        print(f"点 ({px}, {py}) 无法判断类型（需进一步分析）")
