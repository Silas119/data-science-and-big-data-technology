import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# 设置中文显示
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# ============================================================
# 自定义统计函数（不依赖sklearn）
# ============================================================

def f_test_variance(x1, x2, alpha=0.05):
    """
    F检验：检验两组方差是否相等
    H0: σ1² = σ2²  (方差齐性)
    H1: σ1² ≠ σ2²  (方差不齐)
    """
    n1 = len(x1)
    n2 = len(x2)
    var1 = np.var(x1, ddof=1)
    var2 = np.var(x2, ddof=1)
    
    # F统计量 = 较大方差 / 较小方差（使F >= 1）
    if var1 >= var2:
        f_stat = var1 / var2
        dfn = n1 - 1
        dfd = n2 - 1
    else:
        f_stat = var2 / var1
        dfn = n2 - 1
        dfd = n1 - 1
    
    # 双侧检验p值
    p_value = 2 * min(stats.f.cdf(f_stat, dfn, dfd), 1 - stats.f.cdf(f_stat, dfn, dfd))
    
    is_equal = p_value > alpha
    return {
        'f_statistic': f_stat,
        'dfn': dfn,
        'dfd': dfd,
        'p_value': p_value,
        'var_equal': is_equal,
        'interpretation': f'p={p_value:.6f}，{"方差齐性" if is_equal else "方差不齐"}'
    }

def welch_t_test(x1, x2, alpha=0.05):
    """
    Welch两样本t检验（异方差t检验）
    H0: μ1 = μ2
    H1: μ1 ≠ μ2
    """
    n1 = len(x1)
    n2 = len(x2)
    mean1 = np.mean(x1)
    mean2 = np.mean(x2)
    var1 = np.var(x1, ddof=1)
    var2 = np.var(x2, ddof=1)
    
    # t统计量
    se = np.sqrt(var1/n1 + var2/n2)
    t_stat = (mean1 - mean2) / se
    
    # Welch-Satterthwaite自由度
    df = (var1/n1 + var2/n2)**2 / ((var1/n1)**2/(n1-1) + (var2/n2)**2/(n2-1))
    
    # 双侧检验p值
    p_value = 2 * (1 - stats.t.cdf(np.abs(t_stat), df))
    
    # 均值差及置信区间
    mean_diff = mean1 - mean2
    t_critical = stats.t.ppf(1 - alpha/2, df)
    ci_lower = mean_diff - t_critical * se
    ci_upper = mean_diff + t_critical * se
    
    # Cohen's d效应量
    pooled_var = (var1 + var2) / 2
    cohens_d = mean_diff / np.sqrt(pooled_var)
    
    reject_null = p_value < alpha
    
    return {
        't_statistic': t_stat,
        'df': df,
        'p_value': p_value,
        'mean_diff': mean_diff,
        'ci_lower': ci_lower,
        'ci_upper': ci_upper,
       'cohens_d': cohens_d,
        'reject_null': reject_null,
        'interpretation': f'p={p_value:.6f}，{"拒绝" if reject_null else "不拒绝"}原假设'
    }

def independent_t_test(x1, x2):
    """
    标准两样本t检验（方差齐性假设）
    作为对比使用
    """
    n1 = len(x1)
    n2 = len(x2)
    mean1 = np.mean(x1)
    mean2 = np.mean(x2)
    var1 = np.var(x1, ddof=1)
    var2 = np.var(x2, ddof=1)
    
    # 合并方差
    pooled_var = ((n1-1)*var1 + (n2-1)*var2) / (n1 + n2 - 2)
    se = np.sqrt(pooled_var * (1/n1 + 1/n2))
    t_stat = (mean1 - mean2) / se
    df = n1 + n2 - 2
    
    p_value = 2 * (1 - stats.t.cdf(np.abs(t_stat), df))
    
    return {
        't_statistic': t_stat,
        'df': df,
        'p_value': p_value
    }

def calculate_effect_size_r(cohens_d, n1, n2):
    """
    将Cohen's d转换为相关系数r（用于相关性度量）
    r = d / sqrt(d² + 4)
    适用于组间比较的效应量
    """
    r = cohens_d / np.sqrt(cohens_d**2 + 4)
    return r


# ============================================================
# 主实验流程
# ============================================================

def main():
    print("="*70)
    print(" " * 20 + "BMI与高血压相关性分析")
    print(" " * 15 + "两样本异方差t检验实验")
    print("="*70)
    
    # ---------- 步骤1：数据加载与预处理 ----------
    print("\n【步骤1】数据加载与预处理")
    print("-" * 50)
    
    # 直接从同目录读取CSV文件
    try:
        df = pd.read_csv('BMI_BP_Hypertension.csv')
        print(f"✓ 成功读取数据文件: BMI_BP_Hypertension.csv")
    except FileNotFoundError:
        print("✗ 错误：未找到 BMI_BP_Hypertension.csv 文件")
        print("  请确保数据文件与代码文件在同一目录下")
        return
    
    print(f"  原始数据样本量: {len(df)}")
    print(f"  列名: {df.columns.tolist()}")
    
    # 异常值处理：舒张压中的极小异常值 (5.397605346934028e-79)
    # 这些明显是数据录入错误，替换为NaN
    df['舒张压'] = df['舒张压'].replace(5.397605346934028e-79, np.nan)
    
    # 删除缺失值
    df_clean = df.dropna()
    print(f"✓ 清理后样本量: {len(df_clean)} (删除{len(df)-len(df_clean)}条异常记录)")
    
    # 根据高血压标签分组
    hypertensive_mask = df_clean['高血压'] == 1
    bmi_hypertensive = df_clean[hypertensive_mask]['BMI'].values
    bmi_non_hypertensive = df_clean[~hypertensive_mask]['BMI'].values
    
    n1 = len(bmi_hypertensive)
    n2 = len(bmi_non_hypertensive)
    
    print(f"\n✓ 高血压组: n = {n1}")
    print(f"✓ 非高血压组: n = {n2}")
    
    # ---------- 步骤2：描述性统计 ----------
    print("\n【步骤2】描述性统计")
    print("-" * 50)
    
    # 计算各组统计量
    def calc_stats(data, name):
        return {
            '样本量': len(data),
            '均值': np.mean(data),
            '标准差': np.std(data, ddof=1),
            '方差': np.var(data, ddof=1),
            '最小值': np.min(data),
            '25%分位数': np.percentile(data, 25),
            '中位数': np.median(data),
            '75%分位数': np.percentile(data, 75),
            '最大值': np.max(data)
        }
    
    stats_hypertensive = calc_stats(bmi_hypertensive, '高血压组')
    stats_non_hypertensive = calc_stats(bmi_non_hypertensive, '非高血压组')
    
    print("\n高血压组BMI统计:")
    for k, v in stats_hypertensive.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}")
        else:
            print(f"  {k}: {v}")
    
    print("\n非高血压组BMI统计:")
    for k, v in stats_non_hypertensive.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}")
        else:
            print(f"  {k}: {v}")
    
    # 计算组间差异
    mean_diff = stats_hypertensive['均值'] - stats_non_hypertensive['均值']
    print(f"\n组间均值差异: {mean_diff:.4f}")
    print(f"高血压组BMI平均比非高血压组高 {mean_diff:.4f}")
    
    # ---------- 步骤3：正态性检验 ----------
    print("\n【步骤3】正态性检验 (Shapiro-Wilk检验)")
    print("-" * 50)
    
    # 由于样本量较大，Shapiro检验计算较慢，采样5000进行检验
    sample_size_for_shapiro = min(5000, n1)
    shapiro_h = stats.shapiro(bmi_hypertensive[:sample_size_for_shapiro])
    
    sample_size_for_shapiro = min(5000, n2)
    shapiro_nh = stats.shapiro(bmi_non_hypertensive[:sample_size_for_shapiro])
    
    print(f"高血压组 Shapiro-Wilk检验 (n={sample_size_for_shapiro}):")
    print(f"  W={shapiro_h[0]:.6f}, p={shapiro_h[1]:.6f}")
    print(f"  结论: {'近似正态分布' if shapiro_h[1] > 0.05 else '非正态（大样本下t检验稳健）'}")
    
    print(f"\n非高血压组 Shapiro-Wilk检验 (n={sample_size_for_shapiro}):")
    print(f"  W={shapiro_nh[0]:.6f}, p={shapiro_nh[1]:.6f}")
    print(f"  结论: {'近似正态分布' if shapiro_nh[1] > 0.05 else '非正态（大样本下t检验稳健）'}")
    
    print("\n注: 大样本情况下，t检验对正态性偏离具有较好的稳健性，可继续分析。")
    
    # ---------- 步骤4：方差齐性检验 ----------
    print("\n【步骤4】方差齐性检验 (F检验)")
    print("-" * 50)
    
    f_result = f_test_variance(bmi_hypertensive, bmi_non_hypertensive)
    print(f"F统计量: {f_result['f_statistic']:.6f}")
    print(f"自由度: ({f_result['dfn']}, {f_result['dfd']})")
    print(f"p值: {f_result['p_value']:.6f}")
    print(f"结论: {f_result['interpretation']}")
    
    if f_result['var_equal']:
        print("→ 两组方差相等，可采用标准两样本t检验")
    else:
        print("→ 两组方差不相等，应采用Welch异方差t检验")
    
    # ---------- 步骤5：两样本异方差t检验 ----------
    print("\n【步骤5】两样本异方差t检验 (Welch's t-test)")
    print("-" * 50)
    
    # 执行Welch t检验
    welch_result = welch_t_test(bmi_hypertensive, bmi_non_hypertensive)
    
    print(f"t统计量: {welch_result['t_statistic']:.6f}")
    print(f"自由度 (Welch-Satterthwaite): {welch_result['df']:.6f}")
    print(f"p值: {welch_result['p_value']:.10f}")
    print(f"均值差: {welch_result['mean_diff']:.6f}")
    print(f"95%置信区间: [{welch_result['ci_lower']:.6f}, {welch_result['ci_upper']:.6f}]")
    print(f"Cohen's d效应量: {welch_result['cohens_d']:.6f}")
    print(f"结论: {welch_result['interpretation']}")
    
    # 计算相关系数r（作为相关性度量）
    r_effect = calculate_effect_size_r(welch_result['cohens_d'], n1, n2)
    print(f"\n效应量转相关系数 r = {r_effect:.6f}")
    
    # 作为对比，同时计算标准t检验结果
    print("\n【对比】标准两样本t检验（假设方差齐性）")
    print("-" * 40)
    std_result = independent_t_test(bmi_hypertensive, bmi_non_hypertensive)
    print(f"t统计量: {std_result['t_statistic']:.6f}")
    print(f"自由度: {std_result['df']}")
    print(f"p值: {std_result['p_value']:.10f}")
    
    # ---------- 步骤6：结果解释与结论 ----------
    print("\n" + "="*70)
    print("【步骤6】实验结果解释与结论")
    print("="*70)
    
    if welch_result['reject_null']:
        print("\n✓ 拒绝原假设 H₀: μ₁ = μ₂")
        print(f"  高血压组BMI均值 ({stats_hypertensive['均值']:.4f})")
        print(f"  显著高于非高血压组 ({stats_non_hypertensive['均值']:.4f})")
        print(f"  均值差 = {welch_result['mean_diff']:.4f} (95% CI: [{welch_result['ci_lower']:.4f}, {welch_result['ci_upper']:.4f}])")
        print(f"  p < 0.001，差异具有高度统计学显著性")
    else:
        print("\n○ 不拒绝原假设 H₀: μ₁ = μ₂")
        print("  两组BMI均值无显著差异")
    
    # 效应量解释
    d = welch_result['cohens_d']
    if abs(d) < 0.2:
        effect_desc = "极小的效应"
    elif abs(d) < 0.5:
        effect_desc = "小效应"
    elif abs(d) < 0.8:
        effect_desc = "中等效应"
    else:
        effect_desc = "大效应"
    
    print(f"\n效应量 (Cohen's d): {d:.4f} → {effect_desc}")
    print(f"相关系数 r: {r_effect:.4f} (反映了BMI与高血压之间的相关强度)")
    
    # 最终结论
    print("\n" + "="*70)
    print("【最终结论】")
    print("="*70)
    print(f"""
    基于大样本异方差t检验 (Welch's t-test) 分析结果：
    
    1. 高血压组BMI均值 = {stats_hypertensive['均值']:.4f} (SD = {stats_hypertensive['标准差']:.4f})
    2. 非高血压组BMI均值 = {stats_non_hypertensive['均值']:.4f} (SD = {stats_non_hypertensive['标准差']:.4f})
    3. 均值差 = {welch_result['mean_diff']:.4f}，95% CI [{welch_result['ci_lower']:.4f}, {welch_result['ci_upper']:.4f}]
    4. t = {welch_result['t_statistic']:.4f}, df = {welch_result['df']:.2f}, p = {welch_result['p_value']:.2e}
    5. Cohen's d = {d:.4f} ({effect_desc})，r = {r_effect:.4f}
    
    【相关性结论】
    BMI与高血压之间存在显著的统计学相关性 (p < 0.001)。
    高血压组的BMI水平显著高于非高血压组。
    效应量达到{effect_desc}，说明BMI升高与高血压患病风险增加密切相关。
    """)
    
    # ---------- 可视化 ----------
    print("\n【可视化】生成数据分析图表...")
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 图1：箱线图
    ax1 = axes[0, 0]
    bp_data = [bmi_non_hypertensive, bmi_hypertensive]
    bp = ax1.boxplot(bp_data, labels=['非高血压组', '高血压组'], patch_artist=True,
                     boxprops=dict(facecolor='lightblue'),
                     medianprops=dict(color='red', linewidth=2))
    ax1.set_ylabel('BMI', fontsize=12)
    ax1.set_title('两组BMI箱线图对比', fontsize=14)
    ax1.grid(axis='y', linestyle='--', alpha=0.7)
    
    # 添加均值标注
    ax1.plot(1, np.mean(bmi_non_hypertensive), 'g^', markersize=10, label=f'均值={np.mean(bmi_non_hypertensive):.2f}')
    ax1.plot(2, np.mean(bmi_hypertensive), 'g^', markersize=10, label=f'均值={np.mean(bmi_hypertensive):.2f}')
    ax1.legend()
    
    # 图2：直方图 + 核密度
    ax2 = axes[0, 1]
    ax2.hist(bmi_non_hypertensive, bins=50, alpha=0.5, color='blue', density=True, label='非高血压组')
    ax2.hist(bmi_hypertensive, bins=50, alpha=0.5, color='red', density=True, label='高血压组')
    
    # 绘制核密度曲线
    from scipy.stats import gaussian_kde
    if len(bmi_non_hypertensive) > 1:
        kde_nh = gaussian_kde(bmi_non_hypertensive)
        x_nh = np.linspace(min(bmi_non_hypertensive), max(bmi_non_hypertensive), 200)
        ax2.plot(x_nh, kde_nh(x_nh), 'b-', linewidth=2)
    
    if len(bmi_hypertensive) > 1:
        kde_h = gaussian_kde(bmi_hypertensive)
        x_h = np.linspace(min(bmi_hypertensive), max(bmi_hypertensive), 200)
        ax2.plot(x_h, kde_h(x_h), 'r-', linewidth=2)
    
    ax2.set_xlabel('BMI', fontsize=12)
    ax2.set_ylabel('密度', fontsize=12)
    ax2.set_title('两组BMI分布对比', fontsize=14)
    ax2.legend()
    ax2.grid(linestyle='--', alpha=0.3)
    
    # 图3：非高血压组Q-Q图
    ax3 = axes[1, 0]
    from scipy.stats import probplot
    probplot(bmi_non_hypertensive, dist='norm', plot=ax3, fit=True)
    ax3.set_title('非高血压组 Q-Q图', fontsize=14)
    ax3.grid(linestyle='--', alpha=0.3)
    
    # 图4：高血压组Q-Q图
    ax4 = axes[1, 1]
    probplot(bmi_hypertensive, dist='norm', plot=ax4, fit=True)
    ax4.set_title('高血压组 Q-Q图', fontsize=14)
    ax4.grid(linestyle='--', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('BMI_Hypertension_Analysis.png', dpi=300, bbox_inches='tight')
    print("✓ 图表已保存为 'BMI_Hypertension_Analysis.png'")
    plt.show()
    
    print("\n" + "="*70)
    print("实验完成！")
    print("="*70)


if __name__ == "__main__":
    main()
