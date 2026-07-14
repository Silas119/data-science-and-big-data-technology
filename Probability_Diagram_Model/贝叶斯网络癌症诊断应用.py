import itertools
import pandas as pd
class BayesianNetwork:
    def __init__(self):
        self.nodes = []
        self.edges = []
        self.parents = {}
        self.cpds = {}

    def add_node(self, node):
        if node not in self.nodes:
            self.nodes.append(node)
            self.parents[node] = []

    def add_edge(self, parent, child):
        self.add_node(parent)
        self.add_node(child)
        self.edges.append((parent, child))
        self.parents[child].append(parent)

    def add_cpd(self, node, cpd):
        """
        cpd 格式：
        {
            parent_assignment_tuple: {
                node_value: probability
            }
        }

        若节点无父节点，则 parent_assignment_tuple 使用空元组 ()
        """
        self.cpds[node] = cpd

    def check_model(self):
        """
        检查：
        1. 每个节点都有 CPD
        2. 每个 CPD 中概率和为 1
        """
        for node in self.nodes:
            if node not in self.cpds:
                return False

            for parent_values, prob_dist in self.cpds[node].items():
                total_prob = sum(prob_dist.values())
                if abs(total_prob - 1.0) > 1e-8:
                    return False

        return True

    def get_probability(self, node, value, assignment):
        """
        获取 P(node=value | parents)
        """
        parent_values = tuple(assignment[parent] for parent in self.parents[node])
        return self.cpds[node][parent_values][value]

    def joint_probability(self, assignment):
        """
        计算联合概率：
        P(P,S,C,X,D)
        """
        prob = 1.0
        for node in self.nodes:
            prob *= self.get_probability(node, assignment[node], assignment)
        return prob

    def query(self, query_node, evidence):
        """
        用枚举推理计算 P(query_node | evidence)
        """
        hidden_nodes = [
            node for node in self.nodes
            if node != query_node and node not in evidence
        ]

        result = {}

        for query_value in [0, 1]:
            total = 0.0

            for hidden_values in itertools.product([0, 1], repeat=len(hidden_nodes)):
                assignment = dict(evidence)
                assignment[query_node] = query_value

                for node, value in zip(hidden_nodes, hidden_values):
                    assignment[node] = value

                total += self.joint_probability(assignment)

            result[query_value] = total

        # 归一化
        norm = sum(result.values())
        for key in result:
            result[key] /= norm

        return result


# 1. 构建贝叶斯网络

bn = BayesianNetwork()

edges = [
    ('P', 'C'),
    ('S', 'C'),
    ('C', 'X'),
    ('C', 'D')
]

for edge in edges:
    bn.add_edge(*edge)


# 2. 定义条件概率表


# Pollution(P)
# P=0: 无污染，P=1: 有污染
bn.add_cpd('P', {
    (): {
        0: 0.9,
        1: 0.1
    }
})

# Smoker(S)
# S=0: 不吸烟，S=1: 吸烟
bn.add_cpd('S', {
    (): {
        0: 0.7,
        1: 0.3
    }
})

# Cancer(C)
# 父节点顺序为：P, S
bn.add_cpd('C', {
    (0, 0): {
        0: 0.05,
        1: 0.95
    },
    (0, 1): {
        0: 0.03,
        1: 0.97
    },
    (1, 0): {
        0: 0.02,
        1: 0.98
    },
    (1, 1): {
        0: 0.01,
        1: 0.99
    }
})

# X-ray(X)
# 父节点为 C
bn.add_cpd('X', {
    (0,): {
        0: 0.9,
        1: 0.1
    },
    (1,): {
        0: 0.2,
        1: 0.8
    }
})

# Dyspnoea(D)
# 父节点为 C
bn.add_cpd('D', {
    (0,): {
        0: 0.65,
        1: 0.35
    },
    (1,): {
        0: 0.3,
        1: 0.7
    }
})


# 3. 模型验证


check_result = bn.check_model()

print("模型验证结果：", check_result)
print("节点集：", bn.nodes)
print("边集：", bn.edges)


# 4. 推理：P(Cancer | Smoke=0)


posterior = bn.query('C', evidence={'S': 0})

result_table = pd.DataFrame({
    'Cancer': ['C=0', 'C=1'],
    'Probability': [posterior[0], posterior[1]]
})

print("\n给定 Smoke=0 时 Cancer 的条件概率分布：")
print(result_table)
