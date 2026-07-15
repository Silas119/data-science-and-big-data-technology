#创建一个多项式类
class PolyNode:
    #初始化方法
    def __init__(self, coef, exp):
        self.coef = coef  # 系数
        self.exp = exp  # 指数
        self.next = None  # 下一个节点指针
'''方法1：创建链表'''
def create_poly(terms):
    """根据元组列表创建多项式链表"""
    dummy = PolyNode(0, 0)
    current = dummy
    for coef, exp in sorted(terms, key=lambda x: x[1]):
        current.next = PolyNode(coef, exp)
        current = current.next
    return dummy.next
'''方法2：相加'''
def add_polynomials(a, b):
    """多项式相加算法"""
    dummy = PolyNode(0, 0)
    current = dummy

    while a and b:
        if a.exp == b.exp:
            sum_coef = a.coef + b.coef
            if sum_coef != 0:
                current.next = PolyNode(sum_coef, a.exp)
                current = current.next
            a = a.next
            b = b.next
        elif a.exp < b.exp:
            current.next = PolyNode(a.coef, a.exp)
            current = current.next
            a = a.next
        else:
            current.next = PolyNode(b.coef, b.exp)
            current = current.next
            b = b.next

    # 追加剩余节点
    current.next = a if a else b
    return dummy.next

'''方法3：输出'''
def print_poly(node):
    """格式化输出多项式"""
    terms = []
    while node:
        if node.exp == 0:
            terms.append(str(node.coef))
        elif node.exp == 1:
            terms.append(f"{node.coef}x")
        else:
            terms.append(f"{node.coef}x^{node.exp}")
        node = node.next
    print(" + ".join(terms) if terms else "0")
'''执行'''
# 创建示例多项式（系数，次数）
poly_a = create_poly([(7, 0), (3, 1), (9, 8), (5, 17)])
poly_b = create_poly([(8, 1), (22, 7), (-9, 8)])
# 执行相加操作
result = add_polynomials(poly_a, poly_b)
# 输出结果
print("A(x): ", end="")
print_poly(poly_a)  # 7 + 3x + 9x^8 + 5x^17
print("B(x): ", end="")
print_poly(poly_b)  # 8x + 22x^7 - 9x^8
print("结果: ", end="")
print_poly(result)  # 7 + 11x + 22x^7 + 5x^17
