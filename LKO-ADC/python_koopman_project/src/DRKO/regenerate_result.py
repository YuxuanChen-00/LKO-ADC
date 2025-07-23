import pandas as pd
import json
from pathlib import Path
import os

# ==============================================================================
# ✅ 唯一需要您修改的地方 ✅
# 请将下面的路径替换为您实际的 Ray Tune 实验结果文件夹路径。
# 路径通常是 ".../LKO_GridSearch_ray_Final/GridSearch_Delay_IMult_Seed"
EXPERIMENT_RESULTS_PATH = Path(
    "/root/autodl-tmp/python_koopman_project/src/DRKO/models/LKO_GridSearch_ray_Final_motion8_update/GridSearch_Delay_IMult_Seed")


# ==============================================================================


def analyze_ray_results(experiment_path: Path):
    """
    加载 Ray Tune 实验文件夹，读取所有有效试验的结果，
    并生成一个按 RMSE 降序排列的 DataFrame。

    Args:
        experiment_path (Path): 指向 Ray Tune 实验主目录的路径。
    """
    print(f"--- 正在从以下路径加载实验结果 ---\n{experiment_path}\n")

    if not experiment_path.is_dir():
        print(f"❌ 错误: 路径不存在或不是一个文件夹: {experiment_path}")
        return

    # 使用 glob 查找所有包含 params.json 的试验文件夹
    # 这是定位到每一个具体试验的可靠方法
    all_param_files = list(experiment_path.glob("**/params.json"))

    if not all_param_files:
        print("❌ 错误: 在指定路径下未找到任何试验文件夹（没有找到 params.json 文件）。请检查路径是否正确。")
        return

    print(f"🔍 找到了 {len(all_param_files)} 个试验文件夹，正在开始解析...")

    all_valid_results = []
    skipped_count = 0

    for param_file_path in all_param_files:
        trial_path = param_file_path.parent
        result_file_path = trial_path / "result.json"

        # 核心检查：确保 result.json 文件存在且内容不为空
        if not result_file_path.exists() or os.path.getsize(result_file_path) == 0:
            # print(f"  - 跳过: {trial_path.name} (result.json 为空或不存在)")
            skipped_count += 1
            continue

        try:
            # result.json 是一个 JSON-lines 文件，每行是一个JSON对象
            # 我们只需要最后一行，因为它包含了最终的性能指标
            with open(result_file_path, 'r', encoding='utf-8') as f:
                last_line = f.readlines()[-1]

            result_data = json.loads(last_line)

            # 读取对应的超参数文件
            with open(param_file_path, 'r', encoding='utf-8') as f:
                param_data = json.load(f)

            # 提取所需信息
            rmse = result_data.get("rmse")
            delay_step = param_data.get("delay_step")
            i_multiplier = param_data.get("i_multiplier")
            seed = param_data.get("seed")

            # 确保所有需要的信息都成功提取
            if all(v is not None for v in [rmse, delay_step, i_multiplier, seed]):
                all_valid_results.append({
                    "delay_step": delay_step,
                    "i_multiplier": i_multiplier,
                    "seed": seed,
                    "rmse": rmse,
                    "trial_path": str(trial_path)  # 添加用户要求的路径信息
                })
            else:
                skipped_count += 1
                # print(f"  - 跳过: {trial_path.name} (数据字段不完整)")


        except (json.JSONDecodeError, IndexError) as e:
            # 捕获解析错误或文件为空行的异常
            skipped_count += 1
            # print(f"  - 跳过: {trial_path.name} (文件损坏: {e})")
            continue

    print(f"\n--- 解析完成 ---")
    print(f"✔️ 成功解析了 {len(all_valid_results)} 个有效试验结果。")
    print(f"⏭️ 跳过了 {skipped_count} 个无效或空的试验。")

    if not all_valid_results:
        print("\n❌ 未能从任何试验中提取有效结果。")
        return

    # 创建 DataFrame
    results_df = pd.DataFrame(all_valid_results)

    # 按照 RMSE 从高到低排序
    results_df.sort_values(by="rmse", ascending=False, inplace=True)

    # 美化和打印表格
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1200)  # 增加宽度以容纳路径
    pd.set_option('display.max_colwidth', None)  # 显示完整路径

    print("\n\n--- 重新生成的最终结果表 (按 RMSE 降序排列) ---")
    print(results_df.to_string(index=False))

    # 保存到新的CSV文件
    save_path = experiment_path.parent / "regenerated_results_table.csv"
    try:
        results_df.to_csv(save_path, index=False, encoding='utf-8')
        print(f"\n\n✅ 最终表格已成功保存至:\n{save_path}")
    except Exception as e:
        print(f"\n\n❌ 保存文件时出错: {e}")


if __name__ == '__main__':
    analyze_ray_results(EXPERIMENT_RESULTS_PATH)