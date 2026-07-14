import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
import math
from fractions import Fraction
rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
rcParams['axes.unicode_minus'] = False
EPSILON = 1e-10
PRECISION = 6
class BinaryEntropy:
    @staticmethod
    def calc_h(p):
        """计算二元信源熵 H(p)"""
        if p < EPSILON or p > 1 - EPSILON:
            return 0.0
        return -p * math.log2(p) - (1 - p) * math.log2(1 - p)
class DiscreteSource:
    def __init__(self):
        self.probabilities = []
        self.symbols = []
        self.entropy = 0.0
    def input_probabilities(self):
        print("\n" + "=" * 40)
        print("离散信源熵计算")
        print("=" * 40)
        try:
            input_str = input("\n一次性输入所有概率（空格分隔，支持 1/2 0.5）：\n")
            prob_str_list = input_str.strip().split()
            if not prob_str_list:
                print("错误：输入不能为空")
                return False
            self.probabilities = []
            self.symbols = []
            total = 0.0
            for idx, s in enumerate(prob_str_list):
                p = float(Fraction(s.strip()))
                if p < 0 or p > 1:
                    print(f"错误：第{idx+1}个概率超出[0,1]范围")
                    return False
                if p > EPSILON:
                    self.probabilities.append(p)
                    self.symbols.append(f"X{idx+1}")
                    total += p
                else:
                    print(f"符号X{idx+1}概率为0，已自动去除")
            if not self.probabilities:
                print("错误：无有效概率")
                return False
            if abs(total - 1.0) > EPSILON:
                print(f"错误：概率总和为{total:.6f}，必须等于1")
                return False
            print(f"\n有效符号个数：{len(self.probabilities)}")
            print("概率验证通过")
            return True
        except Exception:
            print("错误：输入格式无效，请输入小数或分数（空格分隔）")
            return False
    def calculate_entropy(self):
        self.entropy = 0.0
        for p in self.probabilities:
            self.entropy -= p * math.log2(p)
    def show_result(self):
        print("\n" + "=" * 40)
        print("计算结果")
        print("-" * 40)
        print(f"{'符号':<10}{'概率':<12}{'熵贡献':<12}")
        print("-" * 40)
        for sym, p in zip(self.symbols, self.probabilities):
            ctr = -p * math.log2(p)
            print(f"{sym:<10}{p:<12.6f}{ctr:<12.6f}")
        print("-" * 40)
        print(f"信源熵 H(X) = {self.entropy:.6f} bit/符号")
def main():
    while True:
        print("\n" + "=" * 36)
        print("1  计算离散信源熵")
        print("2  绘制二元信源熵曲线")
        print("3  输入p，输出二元熵 H(p)")
        print("4  退出程序")
        print("=" * 36)
        choice = input("请选择功能：").strip()
        if choice == "1":
            src = DiscreteSource()
            if src.input_probabilities():
                src.calculate_entropy()
                src.show_result()
        elif choice == "2":
            p = np.linspace(0, 1, 1000)
            p_safe = np.clip(p, EPSILON, 1 - EPSILON)
            h = -p_safe * np.log2(p_safe) - (1 - p_safe) * np.log2(1 - p_safe)
            plt.figure(figsize=(10, 6))
            plt.plot(p, h, linewidth=3)
            plt.grid(linestyle='--', alpha=0.3)
            plt.xlabel("符号概率 p")
            plt.ylabel("信源熵 H(p)")
            plt.title("二元对称信源熵曲线")
            plt.tight_layout()
            plt.show()
        elif choice == "3":
            print("\n" + "=" * 30)
            print("计算二元信源熵 H(p)")
            print("=" * 30)
            try:
                p_str = input("请输入概率 p（支持 0.5 或 1/2）：").strip()
                p = float(Fraction(p_str))
                if p < 0 or p > 1:
                    print("错误：p 必须在 [0,1] 范围内")
                else:
                    h_val = BinaryEntropy.calc_h(p)
                    print(f"\np = {p:.6f}")
                    print(f"H(p) = {h_val:.6f} bit/符号")
            except Exception:
                print("错误：输入格式无效")
        elif choice == "4":
            print("程序已退出")
            break
        else:
            print("输入无效，请重新选择")
if __name__ == "__main__":
    main()
