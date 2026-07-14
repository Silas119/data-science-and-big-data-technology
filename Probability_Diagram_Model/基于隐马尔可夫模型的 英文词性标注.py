import os
import math
from collections import defaultdict, Counter
import nltk
from nltk.corpus.reader import BracketParseCorpusReader


# 读取数据（兼容NLTK内置 + 本地文件）
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TREEBANK_DIR = os.path.join(BASE_DIR, "treebank")
nltk.data.path.append(BASE_DIR)

def load_treebank():

        reader = BracketParseCorpusReader(
            root=TREEBANK_DIR,
            fileids=r".*\.mrg"
        )
        return reader.tagged_sents()

tagged_sents = list(load_treebank())

train_sents = tagged_sents[:200]
test_sents = tagged_sents[200:250]

# 构建词表 & 标签集

tags = set()
words = set()

for sent in train_sents:
    for w, t in sent:
        words.add(w)
        tags.add(t)

tags = list(tags)
words = list(words)

tag2id = {t:i for i,t in enumerate(tags)}
word2id = {w:i for i,w in enumerate(words)}

N = len(tags)   # 标签总数
M = len(words)  # 词汇总数


# HMM 参数估计 + 拉普拉斯平滑

pi = defaultdict(float)               # 初始概率
A = defaultdict(lambda: defaultdict(float))  # 转移概率
B = defaultdict(lambda: defaultdict(float))  # 发射概率

pi_count = Counter()
A_count = defaultdict(Counter)
B_count = defaultdict(Counter)
tag_count = Counter()

for sent in train_sents:
    prev_tag = None
    for i, (w, t) in enumerate(sent):
        tag_count[t] += 1
        B_count[t][w] += 1
        if i == 0:
            pi_count[t] += 1
        else:
            A_count[prev_tag][t] += 1
        prev_tag = t

# 拉普拉斯平滑 + 归一化
for t in tags:
    pi[t] = (pi_count[t] + 1) / (len(train_sents) + N)

for t1 in tags:
    total_trans = sum(A_count[t1].values()) + N
    for t2 in tags:
        A[t1][t2] = (A_count[t1][t2] + 1) / total_trans

for t in tags:
    total_emit = sum(B_count[t].values()) + M
    for w in words:
        B[t][w] = (B_count[t][w] + 1) / total_emit

# 1. 参数维度与形状说明

print("HMM 核心参数维度说明")

print(f"标签集大小 N = {N}")
print(f"词汇表大小 M = {M}\n")

print(f"1. 初始概率 Pi")
print(f"   - 形状：一维向量 ({N},)")
print(f"   - 含义：Pi[tag] = P(句首标签为tag)\n")

print(f"2. 转移概率矩阵 A")
print(f"   - 形状：二维矩阵 ({N}, {N})")
print(f"   - 含义：A[prev_tag][curr_tag] = P(curr_tag | prev_tag)\n")

print(f"3. 发射概率矩阵 B")
print(f"   - 形状：二维矩阵 ({N}, {M})")
print(f"   - 含义：B[tag][word] = P(word | tag)")

# 2. 归一化验证

def check_normalization():
    print("\n" + "=" * 60)
    print("归一化验证（理论上每行/向量和应为 1.0）")
    print("=" * 60)

    # 校验初始概率 Pi
    pi_sum = sum(pi.values())
    pi_err = abs(pi_sum - 1.0)
    print(f" 初始概率 Pi 总和 = {pi_sum:.8f}")
    print(f"   与理论值 1.0 的偏差 = {pi_err:.10f}\n")

    # 校验转移矩阵 A：每行和为1
    a_row_sums = [sum(A[t1].values()) for t1 in tags]
    a_max = max(a_row_sums)
    a_min = min(a_row_sums)
    a_max_err = max(abs(s - 1.0) for s in a_row_sums)
    print(f" 转移矩阵 A 行和最大值 = {a_max:.8f}")
    print(f"   转移矩阵 A 行和最小值 = {a_min:.8f}")
    print(f"   行和与 1.0 的最大偏差 = {a_max_err:.10f}\n")

    # 校验发射矩阵 B：每行和为1
    b_row_sums = [sum(B[t].values()) for t in tags]
    b_max = max(b_row_sums)
    b_min = min(b_row_sums)
    b_max_err = max(abs(s - 1.0) for s in b_row_sums)
    print(f" 发射矩阵 B 行和最大值 = {b_max:.8f}")
    print(f"   发射矩阵 B 行和最小值 = {b_min:.8f}")
    print(f"   行和与 1.0 的最大偏差 = {b_max_err:.10f}\n")

    # 最终结论
    all_pass = pi_err < 1e-6 and a_max_err < 1e-6 and b_max_err < 1e-6
    status = " 全部通过" if all_pass else " 存在异常"
    print(f"校验结论：{status}（浮点数精度范围内近似等于 1.0）")

check_normalization()


# Viterbi算法
def viterbi(sentence):
    T = len(sentence)
    dp = [{} for _ in range(T)]
    path = {}

    # 初始化
    for t in tags:
        w = sentence[0]
        emit_prob = B[t].get(w, 1e-8)
        dp[0][t] = math.log(pi[t]) + math.log(emit_prob)
        path[t] = [t]

    # 动态规划递推
    for i in range(1, T):
        new_path = {}
        w = sentence[i]
        for curr in tags:
            best_prob = -1e9
            best_prev = None
            for prev in tags:
                emit_prob = B[curr].get(w, 1e-8)
                prob = dp[i-1][prev] + math.log(A[prev][curr]) + math.log(emit_prob)
                if prob > best_prob:
                    best_prob = prob
                    best_prev = prev
            dp[i][curr] = best_prob
            new_path[curr] = path[best_prev] + [curr]
        path = new_path

    best_tag = max(dp[-1], key=dp[-1].get)
    return path[best_tag]

# 测试预测
y_true = []
y_pred = []

print("示例预测（前3句）")


for i, sent in enumerate(test_sents[:3]):
    words_sent = [w for w, t in sent]
    true_tags = [t for w, t in sent]
    pred_tags = viterbi(words_sent)

    y_true.extend(true_tags)
    y_pred.extend(pred_tags)

    print(f"\nSentence {i+1}:")
    print("Words:", words_sent)
    print("True :", true_tags)
    print("Pred :", pred_tags)

# 评估指标

def evaluate(y_true, y_pred):
    correct = sum(1 for a, b in zip(y_true, y_pred) if a == b)
    acc = correct / len(y_true)

    tag_metrics = {}
    for t in tags:
        TP = sum(1 for a, b in zip(y_true, y_pred) if a == t and b == t)
        FP = sum(1 for a, b in zip(y_true, y_pred) if a != t and b == t)
        FN = sum(1 for a, b in zip(y_true, y_pred) if a == t and b != t)

        precision = TP / (TP + FP + 1e-8)
        recall = TP / (TP + FN + 1e-8)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)
        tag_metrics[t] = (precision, recall, f1)

    macro_f1 = sum(v[2] for v in tag_metrics.values()) / len(tags)

    print("\n" + "=" * 60)
    print("模型评估")
    print("=" * 60)
    print(f"Accuracy  : {acc:.4f}")
    print(f"Macro-F1  : {macro_f1:.4f}")

evaluate(y_true, y_pred)
