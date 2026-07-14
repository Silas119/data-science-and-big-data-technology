import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris

# ==================== 1. 数据集加载 ====================
print("="*60)
print("1. Loading Dataset")
print("="*60)

iris = load_iris()
column_names = ['SepalLength', 'SepalWidth', 'PetalLength', 'PetalWidth', 'Species']
df = pd.DataFrame(data=iris.data, columns=column_names[:-1])
df['Species'] = iris.target_names[iris.target]

print("Dataset shape:", df.shape)
print(df.head())

# ==================== 2. 二分类标签 ====================
print("\n" + "="*60)
print("2. Preprocessing & Binary Label")
print("="*60)

df['BinaryLabel'] = (df['Species'] == 'setosa').astype(int)
X = df.iloc[:, :-2].values
y = df['BinaryLabel'].values

# ==================== 3. 划分数据集 ====================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# ==================== 4. 训练模型 ====================
gnb = GaussianNB()
gnb.fit(X_train, y_train)

# ==================== 5. 预测 ====================
y_pred = gnb.predict(X_test)
y_pred_proba = gnb.predict_proba(X_test)
cm = confusion_matrix(y_test, y_pred)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) != 0 else 0

# ==================== 6. 四张图分别保存（英文标题） ====================
print("\n" + "="*60)
print("Saving 4 separate plots...")
print("="*60)

# -------------------- 图1：混淆矩阵 --------------------
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Negative', 'Positive'],
            yticklabels=['Negative', 'Positive'])
plt.title('Confusion Matrix', fontweight='bold')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.tight_layout()
plt.savefig('plot_1_confusion_matrix.png', dpi=300)
plt.close()

# -------------------- 图2：评估指标柱状图 --------------------
plt.figure(figsize=(6, 5))
metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
values = [accuracy, precision, recall, f1]
colors = ['#2ecc71', '#3498db', '#e74c3c', '#f39c12']
plt.bar(metrics, values, color=colors, alpha=0.7, edgecolor='black')
plt.ylim(0, 1.05)
plt.ylabel('Score')
plt.title('Model Evaluation Metrics', fontweight='bold')
for i, v in enumerate(values):
    plt.text(i, v + 0.02, f'{v:.4f}', ha='center', fontweight='bold')
plt.tight_layout()
plt.savefig('plot_2_metrics.png', dpi=300)
plt.close()

# -------------------- 图3：测试集类别分布 --------------------
plt.figure(figsize=(6, 5))
class_counts = [np.sum(y_test==0), np.sum(y_test==1)]
plt.bar(['Negative', 'Positive'], class_counts, color=['#e74c3c', '#2ecc71'], alpha=0.7, edgecolor='black')
plt.ylabel('Number of Samples')
plt.title('Test Set Class Distribution', fontweight='bold')
for i, c in enumerate(class_counts):
    plt.text(i, c + 0.2, str(int(c)), ha='center', fontweight='bold')
plt.tight_layout()
plt.savefig('plot_3_class_distribution.png', dpi=300)
plt.close()

# -------------------- 图4：预测概率分布 --------------------
plt.figure(figsize=(6, 5))
plt.hist(y_pred_proba[y_test==0, 1], bins=15, alpha=0.6, label='Negative Class', color='#e74c3c')
plt.hist(y_pred_proba[y_test==1, 1], bins=15, alpha=0.6, label='Positive Class', color='#2ecc71')
plt.axvline(0.5, color='black', linestyle='--', linewidth=2, label='Decision Boundary')
plt.xlabel('Positive Class Probability')
plt.ylabel('Frequency')
plt.title('Predicted Probability Distribution', fontweight='bold')
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('plot_4_prob_distribution.png', dpi=300)
plt.close()

print("\n 4 images saved successfully:")
print("1. plot_1_confusion_matrix.png")
print("2. plot_2_metrics.png")
print("3. plot_3_class_distribution.png")
print("4. plot_4_prob_distribution.png")
