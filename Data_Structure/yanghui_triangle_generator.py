def print_pascal_triangle(n):
    current_row = [1]
    for _ in range(n):
        # 打印当前行
        print(' '.join(map(str, current_row)).center(n * 2))
        # 生成下一行
        temp_row = list(current_row)
        sum_elements = []
        if temp_row:
            prev = temp_row.pop(0)
            while temp_row:
                current = temp_row.pop(0)
                sum_elements.append(prev + current)
                prev = current
        # 构建下一行
        next_row = [1]
        next_row.extend(sum_elements)
        next_row.append(1)
        current_row = next_row

# 示例：打印5行杨辉三角
print_pascal_triangle(5)
