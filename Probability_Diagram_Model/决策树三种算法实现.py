import os
import math
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from matplotlib.font_manager import FontProperties

# 图片保存到当前源代码同一文件夹
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False
#1、表格数据的转化
data = [
    [1, "青年", "否", "否", "一般", "否"],
    [2, "青年", "否", "否", "好", "否"],
    [3, "青年", "是", "否", "好", "是"],
    [4, "青年", "是", "是", "一般", "是"],
    [5, "青年", "否", "否", "一般", "否"],
    [6, "中年", "否", "否", "一般", "否"],
    [7, "中年", "否", "否", "好", "否"],
    [8, "中年", "是", "是", "好", "是"],
    [9, "中年", "否", "是", "非常好", "是"],
    [10, "中年", "否", "是", "非常好", "是"],
    [11, "老年", "否", "是", "非常好", "是"],
    [12, "老年", "否", "是", "好", "是"],
    [13, "老年", "是", "否", "好", "是"],
    [14, "老年", "是", "否", "非常好", "是"],
    [15, "老年", "否", "否", "一般", "否"],
]

df = pd.DataFrame(
    data,
    columns=["ID", "年龄", "有工作", "有自己的房子", "信贷情况", "类别"]
)

features = ["年龄", "有工作", "有自己的房子", "信贷情况"]

#2、决策树训练
def entropy(labels):
    counts = labels.value_counts()
    total = len(labels)
    return -sum((c / total) * math.log2(c / total) for c in counts)


def majority(labels):
    return labels.value_counts().idxmax()


def info_gain(df, feature):
    h = entropy(df["类别"])
    cond_entropy = 0
    for _, sub_df in df.groupby(feature):
        cond_entropy += len(sub_df) / len(df) * entropy(sub_df["类别"])
    return h - cond_entropy


def gain_ratio(df, feature):
    gain = info_gain(df, feature)
    split_info = entropy(df[feature])
    return gain / split_info if split_info != 0 else 0


def gini(labels):
    counts = labels.value_counts()
    total = len(labels)
    return 1 - sum((c / total) ** 2 for c in counts)


def gini_index(df, feature):
    total = len(df)
    return sum(len(sub_df) / total * gini(sub_df["类别"])
               for _, sub_df in df.groupby(feature))


def build_tree(df, features, method):
    labels = df["类别"]

    if len(labels.unique()) == 1:
        return labels.iloc[0]

    if len(features) == 0:
        return majority(labels)

    if method == "ID3":
        best_feature = max(features, key=lambda f: info_gain(df, f))
    elif method == "C4.5":
        best_feature = max(features, key=lambda f: gain_ratio(df, f))
    elif method == "CART":
        best_feature = min(features, key=lambda f: gini_index(df, f))
    else:
        raise ValueError("method must be ID3, C4.5 or CART")

    tree = {best_feature: {}}
    remaining_features = [f for f in features if f != best_feature]

    for value, sub_df in df.groupby(best_feature):
        tree[best_feature][value] = build_tree(
            sub_df.drop(columns=[best_feature]),
            remaining_features,
            method
        )

    return tree

#3、决策树可视化与图片保存
def leaf_count(tree):
    if not isinstance(tree, dict):
        return 1
    root = next(iter(tree))
    return sum(leaf_count(child) for child in tree[root].values())


def draw_tree(tree, title, filename):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.axis("off")

    def draw_node(x, y, text):
        box = FancyBboxPatch(
            (x - 0.11, y - 0.05),
            0.22,
            0.10,
            boxstyle="round,pad=0.02",
            linewidth=1.2,
            facecolor="white"
        )
        ax.add_patch(box)
        ax.text(x, y, text, ha="center", va="center", fontsize=12)

    def draw_subtree(subtree, x0, x1, y, dy):
        x = (x0 + x1) / 2

        if not isinstance(subtree, dict):
            draw_node(x, y, subtree)
            return x

        root = next(iter(subtree))
        draw_node(x, y, root)

        total_leaf = sum(leaf_count(child) for child in subtree[root].values())
        cur = x0

        for edge_value, child in subtree[root].items():
            count = leaf_count(child)
            nx0 = cur
            nx1 = cur + (x1 - x0) * count / total_leaf
            child_x = draw_subtree(child, nx0, nx1, y - dy, dy)

            ax.plot([x, child_x], [y - 0.06, y - dy + 0.06], linewidth=1)
            ax.text(
                (x + child_x) / 2,
                (y + y - dy) / 2 + 0.02,
                str(edge_value),
                ha="center",
                va="center",
                fontsize=11,
                bbox=dict(facecolor="white", edgecolor="none", pad=1)
            )

            cur = nx1

        return x

    draw_subtree(tree, 0.08, 0.92, 0.86, 0.28)

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title(title, fontsize=15)

    save_path = os.path.join(BASE_DIR, filename)
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()

    print(f"{title} 已保存：{save_path}")


id3_tree = build_tree(df, features, "ID3")
c45_tree = build_tree(df, features, "C4.5")
cart_tree = build_tree(df, features, "CART")

print("ID3决策树：", id3_tree)
print("C4.5决策树：", c45_tree)
print("CART决策树：", cart_tree)

draw_tree(id3_tree, "图1 ID3算法生成的决策树", "图1_ID3算法生成的决策树.png")
draw_tree(c45_tree, "图2 C4.5算法生成的决策树", "图2_C45算法生成的决策树.png")
draw_tree(cart_tree, "图3 CART算法生成的决策树", "图3_CART算法生成的决策树.png")
