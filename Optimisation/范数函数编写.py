import numpy as np
#函数1
def vector_norm(v, norm_type=2):
    """计算向量的1-范数、2-范数或无穷范数"""
    v = np.asarray(v)
    if v.ndim != 1:
        raise ValueError("输入必须是一维向量")

    if norm_type == 1:
        return np.linalg.norm(v, ord=1)
    elif norm_type == 2:
        return np.linalg.norm(v, ord=2)
    elif norm_type == 'inf':
        return np.linalg.norm(v, ord=np.inf)
    else:
        raise ValueError("只支持 1, 2, 'inf' 三种范数类型")

#函数2
def matrix_norm(mat, norm_type=2):
    """计算矩阵的F-范数(2-范数)、1-范数和无穷范数"""
    mat = np.asarray(mat)
    if mat.ndim != 2:
        raise ValueError("输入必须是二维矩阵")
 # 1-范数（列范数）：列绝对值之和的最大值
    if norm_type == 1:
        return np.linalg.norm(mat, ord=1)
# 2-范数（F-范数）：所以元素平方和开根号
    elif norm_type == 2:
        return np.linalg.norm(mat, ord='fro')  
# 无穷范数（行范数）：行绝对值之和的最大值
    elif norm_type == 'inf':
        return np.linalg.norm(mat, ord=np.inf)
    else:
        raise ValueError("只支持 1, 2, 'inf' 三种范数类型")

#函数3
def matrix_operator_norm(mat, norm_type=2):
    """计算矩阵的算子范数(1-范数、2-范数、无穷范数)"""
    mat = np.asarray(mat)
    if mat.ndim != 2:
        raise ValueError("输入必须是二维矩阵")
# 1-范数（列范数）
    if norm_type == 1:
        return np.linalg.norm(mat, ord=1)
# 2-范数：最大奇异值 
    elif norm_type == 2:
        return np.linalg.norm(mat, ord=2)  
#无穷范数（行范数）
    elif norm_type == 'inf':
        return np.linalg.norm(mat, ord=np.inf)
    else:
        raise ValueError("只支持 1, 2, 'inf' 三种范数类型")


def main():
    # 设置随机种子
    np.random.seed(42)

    # 随机生成一个向量
    vec = np.random.randn(5)  
    print("随机生成的向量:")
    print(vec)
    print("\n向量范数计算结果:")
    print(f"1-范数: {vector_norm(vec, 1):.4f}")
    print(f"2-范数: {vector_norm(vec, 2):.4f}")
    print(f"无穷范数: {vector_norm(vec, 'inf'):.4f}\n")

    # 随机生成一个矩阵
    mat = np.random.randint(-10, 10, size=(3, 4))  
    print("随机生成的矩阵:")
    print(mat)
    print("\n矩阵范数(包括F-范数)计算结果:")
    print(f"1-范数: {matrix_norm(mat, 1):.4f}")
    print(f"F-范数(2-范数): {matrix_norm(mat, 2):.4f}")
    print(f"无穷范数: {matrix_norm(mat, 'inf'):.4f}")

    print("\n矩阵算子范数计算结果:")
    print(f"1-范数(列范数): {matrix_operator_norm(mat, 1):.4f}")
    print(f"2-范数(谱范数): {matrix_operator_norm(mat, 2):.4f}")
    print(f"无穷范数(行范数): {matrix_operator_norm(mat, 'inf'):.4f}")


if __name__ == "__main__":
    main()
