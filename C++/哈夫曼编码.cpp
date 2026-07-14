#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define MAX_NODES 100
#define MAX_CODE_LEN 100

// 数据结构设计

// 哈夫曼树节点
typedef struct {
    double weight;      // 概率(权值)
    int parent;         // 父节点下标
    int lchild;         // 左孩子下标
    int rchild;         // 右孩子下标
    int original_index; // 排序前的原始索引(用于追踪符号)
} HNode;

// 编码结果存储
typedef struct {
    double weight;
    char bits[MAX_CODE_LEN];
    int len;
} HCode;

// --- 函数声明 ---
void run_experiment(double* probs, int n, char* title);
void sort_probabilities(double* probs, int n);
int check_probabilities(double* probs, int n);

// --- 主函数 ---
int main() {
    // 习题 5-10 数据
    double probs_5_10[] = { 0.37, 0.25, 0.18, 0.10, 0.07, 0.03 };
    int n_5_10 = sizeof(probs_5_10) / sizeof(probs_5_10[0]);

    // 习题 5-11 数据
    double probs_5_11[] = { 0.32, 0.22, 0.18, 0.16, 0.08, 0.04 };
    int n_5_11 = sizeof(probs_5_11) / sizeof(probs_5_11[0]);

    printf("哈夫曼编码实验 \n\n");

    // 运行 5-10
    run_experiment(probs_5_10, n_5_10, "习题 5-10");


    // 运行 5-11
    run_experiment(probs_5_11, n_5_11, "习题 5-11");

    return 0;
}

// --- 功能实现 ---

// 1. 检验概率和
int check_probabilities(double* probs, int n) {
    double sum = 0.0;
    for (int i = 0; i < n; i++) sum += probs[i];
    // 浮点数比较，允许微小误差
    return (fabs(sum - 1.0) < 1e-5);
}

// 2. 降序排序 (简单的冒泡排序)
void sort_probabilities(double* probs, int n) {
    for (int i = 0; i < n - 1; i++) {
        for (int j = 0; j < n - 1 - i; j++) {
            if (probs[j] < probs[j + 1]) { // 降序
                double temp = probs[j];
                probs[j] = probs[j + 1];
                probs[j + 1] = temp;
            }
        }
    }
}

// 核心逻辑封装
void run_experiment(double* input_probs, int n, char* title) {
    printf("[%s] 开始处理...\n", title);

    // 步骤 1: 检验概率
    if (!check_probabilities(input_probs, n)) {
        printf("错误：概率之和不为 1！\n");
        return;
    }
    else {
        printf("1. 概率检验: 通过 (Sum = 1.0)\n");
    }

    // 步骤 2: 降序排列
    // 注意：为了不破坏原数组用于对比，这里拷贝一份
    double probs[MAX_NODES];
    for (int i = 0; i < n; i++) probs[i] = input_probs[i];
    sort_probabilities(probs, n);

    printf("2. 降序排列后: ");
    for (int i = 0; i < n; i++) printf("%.2f ", probs[i]);
    printf("\n");

    // 步骤 3: 构建哈夫曼树
    HNode huffTree[2 * MAX_NODES - 1];
    HCode huffCodes[MAX_NODES];

    // 初始化叶子节点
    for (int i = 0; i < 2 * n - 1; i++) {
        huffTree[i].parent = -1;
        huffTree[i].lchild = -1;
        huffTree[i].rchild = -1;
        if (i < n) {
            huffTree[i].weight = probs[i];
            huffTree[i].original_index = i;
        }
        else {
            huffTree[i].weight = 0.0;
        }
    }

    // 构造过程：寻找两个最小的节点合并
    for (int i = 0; i < n - 1; i++) {
        int m1 = -1, m2 = -1;
        double min1 = 100.0, min2 = 100.0;

        // 扫描所有节点，找出没有父节点且权值最小的两个
        for (int j = 0; j < n + i; j++) {
            if (huffTree[j].parent == -1) {
                if (huffTree[j].weight < min1) {
                    min2 = min1;
                    m2 = m1;
                    min1 = huffTree[j].weight;
                    m1 = j;
                }
                else if (huffTree[j].weight < min2) {
                    min2 = huffTree[j].weight;
                    m2 = j;
                }
            }
        }

        // 合并 m1 和 m2 生成新节点
        int newNodeIndex = n + i;
        huffTree[m1].parent = newNodeIndex;
        huffTree[m2].parent = newNodeIndex;
        huffTree[newNodeIndex].weight = min1 + min2;
        huffTree[newNodeIndex].lchild = m1; // 较小的作为左孩子
        huffTree[newNodeIndex].rchild = m2; // 较大的作为右孩子
    }

    // 步骤 4: 求出码字 (从叶子向上回溯)
    printf("3. 哈夫曼编码结果:\n");
    printf("   符号(P) \t 码字 \t\t 码长\n");
    printf("   ------- \t ---- \t\t ----\n");

    double avg_len = 0.0;
    double entropy = 0.0;

    for (int i = 0; i < n; i++) {
        int current = i;
        int parent = huffTree[current].parent;
        int start = 0;
        char tempBits[MAX_CODE_LEN];

        // 向上回溯
        while (parent != -1) {
            if (huffTree[parent].lchild == current) {
                tempBits[start++] = '0'; // 左孩子编码 0
            }
            else {
                tempBits[start++] = '1'; // 右孩子编码 1
            }
            current = parent;
            parent = huffTree[current].parent;
        }

        // 反转字符串得到正确编码
        huffCodes[i].len = start;
        huffCodes[i].weight = probs[i];
        for (int j = 0; j < start; j++) {
            huffCodes[i].bits[j] = tempBits[start - 1 - j];
        }
        huffCodes[i].bits[start] = '\0';

        // 打印结果
        printf("   %.2f    \t %-10s \t %d\n", probs[i], huffCodes[i].bits, huffCodes[i].len);

        // 累加计算指标
        avg_len += probs[i] * huffCodes[i].len;
        if (probs[i] > 0) {
            entropy += -1 * probs[i] * log2(probs[i]);
        }
    }

    // 步骤 5: 计算指标
    double efficiency = (entropy / avg_len) * 100.0;

    printf("\n4. 性能指标:\n");
    printf("   信源熵 H(X)   = %.4f bit/符号\n", entropy);
    printf("   平均码长 L    = %.4f bit/符号\n", avg_len);
    printf("   编码效率 eta  = %.2f%%\n", efficiency);
}
