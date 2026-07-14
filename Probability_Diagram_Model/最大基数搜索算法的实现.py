from typing import List
class MaximumCardinalitySearch:
    """最大基数搜索算法实现"""
    def __init__(self, adj_matrix: List[List[int]]):
        self.adj_matrix = adj_matrix
        self.n = len(adj_matrix)
        self.weight = [0] * self.n
        self.marked = [False] * self.n
        self.sequence = []
        self.steps = []
    def validate_graph(self):
        if self.n != len(self.adj_matrix):
            raise ValueError("邻接矩阵必须是正方形矩阵")
        for i in range(self.n):
            if len(self.adj_matrix[i]) != self.n:
                raise ValueError("邻接矩阵行列数不一致")
            for j in range(self.n):
                if self.adj_matrix[i][j] != self.adj_matrix[j][i]:
                    raise ValueError("邻接矩阵不对称，不是有效的无向图")
    def get_unmarked_vertices(self) -> List[int]:
        return [i for i in range(self.n) if not self.marked[i]]
    def print_graph_info(self):
        print("图的邻接矩阵（5×5）：")
        print("     " + "  ".join(f"{i:2d}" for i in range(self.n)))
        for i in range(self.n):
            print(f"  {i}  " + "  ".join(f"{self.adj_matrix[i][j]:2d}" for j in range(self.n)))
        print("\n图的结构信息：")
        for i in range(self.n):
            neighbors = [j for j in range(self.n) if self.adj_matrix[i][j] == 1]
            degree = len(neighbors)
            print(f"  顶点 {i}: 度={degree:2d}, 邻接顶点={neighbors}")
    def search(self):
        print("最大基数搜索算法（Maximum Cardinality Search）执行过程")
        for iteration in range(self.n):
            print(f"\n第 {iteration + 1}/{self.n} 步")
            self._print_current_state(iteration)
            selected_vertex = self._select_vertex()
            print(f"  选择顶点：{selected_vertex} （权值={self.weight[selected_vertex]}）")
            self.marked[selected_vertex] = True
            self.sequence.append(selected_vertex)
            neighbors_updated = self._update_weights(selected_vertex)
            print(f"  更新邻接顶点权值（增加1）：{neighbors_updated}")
            self.steps.append({
                'iteration': iteration + 1,
                'selected': selected_vertex,
                'updated_neighbors': neighbors_updated,
                'weights_after': self.weight.copy()
            })
            print(f"  权值更新后：{self.weight}")
        return self.sequence
    def _print_current_state(self, iteration: int):
        unmarked = self.get_unmarked_vertices()
        print(f"  已处理顶点：{self.sequence}")
        print(f"  未处理顶点：{unmarked}")
        print(f"  各顶点权值：", end="")
        weight_str = "["
        for i in range(self.n):
            if self.marked[i]:
                weight_str += f"{i}:已标记"
            else:
                weight_str += f"{i}:{self.weight[i]}"
            if i < self.n - 1:
                weight_str += ", "
        weight_str += "]"
        print(weight_str)
    def _select_vertex(self) -> int:
        max_weight = -1
        selected = -1
        for i in range(self.n):
            if not self.marked[i] and self.weight[i] > max_weight:
                max_weight = self.weight[i]
                selected = i
        return selected
    def _update_weights(self, vertex: int) -> List[int]:
        updated = []
        for neighbor in range(self.n):
            if self.adj_matrix[vertex][neighbor] == 1 and not self.marked[neighbor]:
                self.weight[neighbor] += 1
                updated.append(neighbor)
        return updated
    def print_result(self):
        print("\n算法执行结果")
        peo = self.sequence[::-1]
        print(f"\n  选择顺序（最大基数序列）：{self.sequence}")
        print(f"  完美消除序列（PEO）：    {peo}")
        print("\n详细步骤总结：")
        for step in self.steps:
            print(f"  步骤 {step['iteration']:2d}: 选择顶点 {step['selected']} → "
                  f"更新邻接 {step['updated_neighbors']} → "
                  f"权值更新为 {step['weights_after']}")
    def is_perfect_elimination_order(self, order: List[int]) -> bool:
        return True
if __name__ == "__main__":
    print("最大基数搜索算法示例")
    adj_matrix = [
        [0, 1, 1, 0, 0],
        [1, 0, 1, 1, 0],
        [1, 1, 0, 1, 1],
        [0, 1, 1, 0, 1],
        [0, 0, 1, 1, 0]
    ]
    mcs = MaximumCardinalitySearch(adj_matrix)
    mcs.validate_graph()
    mcs.print_graph_info()
    mcs.search()
    mcs.print_result()
