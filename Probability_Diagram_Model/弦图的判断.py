import networkx as nx
# 构建图（顶点编号1-5，边根据图片添加）
G = nx.Graph()
G.add_nodes_from([1, 2, 3, 4, 5])
G.add_edges_from([(1,2), (1,3), (1,4),
                  (2,3), (2,5),
                  (3,4), (3,5),
                  (4,5)])
# 直接使用networkx判断是否为弦图
is_chordal = nx.is_chordal(G)
print("是否为弦图:", is_chordal)  # 输出: False
# 进一步验证：找出所有长度≥4的基圈（cycle basis），并检查是否存在无弦的圈
cycles = nx.cycle_basis(G)
print("图中的基圈（长度≥3）:", cycles)
# 重点检查长度为4的圈是否都有弦
for cycle in cycles:
    if len(cycle) >= 4:
        # 检查该圈是否有弦（非相邻顶点间是否有边）
        has_chord = False
        for i in range(len(cycle)):
            u = cycle[i]
            v = cycle[(i+2) % len(cycle)]  # 跳过相邻，检查对角线
            if G.has_edge(u, v):
                has_chord = True
                break
        if not has_chord:
            print(f"发现无弦的{len(cycle)}-圈: {cycle}")
            break
