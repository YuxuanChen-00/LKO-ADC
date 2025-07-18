import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


def create_errorbar_plots():
    """
    加载聚合后的实验结果，并为每个delay_step生成
    i_multiplier vs. RMSE 的误差棒图。
    """
    # --- 1. 定义文件路径 ---
    # 脚本会自动查找上一步生成的 a-generated_results_table.csv 文件
    base_path = Path(__file__).resolve().parent
    results_csv_path = base_path / "models" / "LKO_GridSearch_ray_Final" / "regenerated_results_table.csv"

    output_plot_dir = base_path / "analysis_plots"
    output_plot_dir.mkdir(exist_ok=True)  # 创建用于存放图片的文件夹

    print(f"--- 正在从以下路径加载结果文件 ---\n{results_csv_path}\n")

    # --- 2. 加载数据 ---
    try:
        df = pd.read_csv(results_csv_path)
    except FileNotFoundError:
        print(f"❌ 错误: 未找到结果文件 '{results_csv_path}'。")
        print("请确保此脚本与您之前的脚本位于同一目录下，或者手动修改脚本中的路径。")
        return

    # --- 3. 准备绘图 ---
    # 设置一个美观的绘图风格
    sns.set_theme(style="whitegrid", palette="viridis")

    # 获取所有独立的 delay_step 值
    unique_delay_steps = sorted(df['delay_step'].unique())

    print(f"🔍 发现 {len(unique_delay_steps)} 个独立的 delay_step 值: {unique_delay_steps}")
    print("--- 开始为每个 delay_step 生成误差棒图 ---")

    generated_files = []

    # --- 4. 循环并为每个 delay_step 绘图 ---
    for delay in unique_delay_steps:
        # a. 筛选出当前 delay_step 的数据
        df_delay = df[df['delay_step'] == delay].copy()

        # b. 按 i_multiplier 分组，计算每个组的 RMSE 均值和标准差
        #    标准差将作为误差棒的范围
        agg_df = df_delay.groupby('i_multiplier')['rmse'].agg(['mean', 'std']).reset_index()
        agg_df.sort_values('i_multiplier', inplace=True)

        # 如果某个(i_multiplier)只有一个seed, 其std会是NaN, 用0填充
        agg_df['std'].fillna(0, inplace=True)

        # c. 创建图形
        plt.figure(figsize=(14, 8))

        errorbar_plot = plt.errorbar(
            x=agg_df['i_multiplier'],
            y=agg_df['mean'],
            yerr=agg_df['std'],
            fmt='-o',  # 'o' 表示数据点, '-' 表示连接线
            capsize=5,  # 误差棒顶端横线的大小
            capthick=2,  # 误差棒顶端横线的粗细
            elinewidth=2,  # 误差棒竖线的粗细
            markersize=8,  # 数据点的大小
            label=f'Mean RMSE (Std Dev across {df_delay["seed"].nunique()} seeds)'
        )

        # d. 美化图形
        plt.title(f'RMSE vs. i_multiplier for delay_step = {delay}', fontsize=18, weight='bold')
        plt.xlabel('i_multiplier (升维乘子)', fontsize=14)
        plt.ylabel('Mean RMSE (跨种子平均值)', fontsize=14)
        plt.xticks(agg_df['i_multiplier'], rotation=45)  # 确保所有x轴刻度都显示
        plt.tick_params(axis='both', which='major', labelsize=12)
        plt.grid(True, which='both', linestyle='--', linewidth=0.5)
        plt.legend(fontsize=12)
        plt.tight_layout()  # 自动调整布局，防止标签重叠

        # e. 保存图形
        plot_filename = f"errorbar_delay_{delay}.png"
        save_path = output_plot_dir / plot_filename
        plt.savefig(save_path, dpi=150)
        plt.close()  # 关闭当前图形，释放内存，为下一张图做准备

        generated_files.append(save_path)
        print(f"✅ 已生成图片: {save_path}")

    print("\n--- 所有绘图任务已完成！---")


if __name__ == '__main__':
    create_errorbar_plots()