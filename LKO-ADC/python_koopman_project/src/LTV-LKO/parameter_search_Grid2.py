# file: parameter_search_hybrid.py

import torch
import numpy as np
from scipy.io import loadmat
import random
from pathlib import Path
import time
import os
import shutil
import ray
from ray import tune
from ray.tune import Checkpoint
import pandas as pd
import json

# --- 1. 导入您当前项目的核心模块 ---
# 确保这些文件在您的项目中且路径正确
from train_hybrid_ltv_model import train_two_stage_hybrid_model, set_global_seed
from dataloader import generate_koopman_data
from src.normalize_data import normalize_data

# --- 全局设置 ---
# 定义网格搜索结果和模型的保存根目录
BASE_RESULTS_PATH = Path(__file__).resolve().parent / "models" / "Hybrid_LTV_GridSearch_Final"


def train_trial(config, data_dict, base_params):
    """
    Ray Tune的单次试验（Trial）执行函数。
    此函数负责根据接收到的一组超参数(config)来执行完整的训练和评估流程。
    """
    # --- 为当前试验设置独立的随机种子 ---
    set_global_seed(config['seed'])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- a. 组合参数 ---
    # 将基础参数与当前试验的超参数(config)合并
    params = base_params.copy()
    params.update(config)
    params['device'] = device
    # 计算依赖于超参数的派生参数
    params['PhiDimensions'] = params['delay_step'] * params['i_multiplier']

    # --- b. 为当前试验动态生成数据 ---
    # 这是必要的，因为 `delay_step` 是一个超参数，它会改变输入数据的形状
    print(
        f"Trial (seed={params['seed']}, delay_step={params['delay_step']}, i_multiplier={params['i_multiplier']}): Generating specific dataset...")

    normalized_train_data = data_dict['normalized_train_data']
    normalized_test_data = data_dict['normalized_test_data']

    # 为当前 delay_step 生成训练数据
    control_train_list, state_train_list, label_train_list = [], [], []
    for raw_data in normalized_train_data:
        ctrl_td, state_td, label_td = generate_koopman_data(
            raw_data['control'], raw_data['state'], params['delay_step'], params['pred_step']
        )
        control_train_list.append(ctrl_td)
        state_train_list.append(state_td)
        label_train_list.append(label_td)

    train_data_full = {
        'control': np.concatenate(control_train_list, axis=0),
        'state': np.concatenate(state_train_list, axis=0),
        'label': np.concatenate(label_train_list, axis=0)
    }

    # 为当前 delay_step 生成测试数据
    test_data_full = []
    for raw_data in normalized_test_data:
        ctrl_td, state_td, label_td = generate_koopman_data(
            raw_data['control'], raw_data['state'], params['delay_step'], params['pred_step']
        )
        test_data_full.append({
            'control': torch.from_numpy(ctrl_td).float().to(device),
            'state': torch.from_numpy(state_td).float().to(device),
            'label': torch.from_numpy(label_td).float().to(device)
        })

    # --- c. 调用核心训练与评估流程 ---
    # `train_two_stage_hybrid_model` 函数内部已包含了完整的训练和评估
    # 它会返回最终训练好的模型和在测试集上的最佳RMSE
    print(f"Trial (seed={params['seed']}, delay_step={params['delay_step']}): Starting training...")
    net, final_rmse = train_two_stage_hybrid_model(params, train_data_full, test_data_full)

    # --- d. 保存检查点并报告结果 ---
    checkpoint = None
    if net is not None and np.isfinite(final_rmse):
        checkpoint_dir = os.path.join(os.getcwd(), "checkpoint")
        os.makedirs(checkpoint_dir, exist_ok=True)
        model_path = os.path.join(checkpoint_dir, "final_model.pth")
        torch.save(net.state_dict(), model_path)
        checkpoint = Checkpoint.from_directory(checkpoint_dir)

    # 向Ray Tune报告本次试验的最终评估指标
    tune.report({"rmse": final_rmse}, checkpoint=checkpoint)
    print(f"Trial (seed={params['seed']}, delay_step={params['delay_step']}): Finished. Final RMSE: {final_rmse:.6f}")


def main():
    """
    主执行函数，负责配置和运行整个网格搜索流程。
    """
    start_time_total = time.time()
    print("## 1. 设置基础参数和路径... ##")
    BASE_RESULTS_PATH.mkdir(parents=True, exist_ok=True)

    # --- 基础参数 (不会在网格搜索中改变的部分) ---
    # 从您的main.py和train_...py中提取
    base_params = {
        'is_norm': True, 'state_size': 6, 'control_size': 6,
        'pred_step': 1,  # 与main.py保持一致
        'batchSize': 1024,
        'minLearnRate': 1e-7,
        # 模型架构
        'encoder_gru_hidden': 256,
        'encoder_mlp_hidden': 128,
        'delta_rnn_hidden': 64,
        'delta_mlp_hidden': 64,
        # 两阶段训练周期和学习率
        'num_epochs_s1': 150,
        'num_epochs_s2': 50,
        'lr_s1': 3.3e-5,
        'lr_s2': 1.0e-6,
        # 损失权重
        'L1': 0.88, 'L2': 0.28, 'L3': 37.28, 'L_delta': 0.62,
    }

    # --- 2. 一次性加载和预处理数据 (与参考文件逻辑一致) ---
    print("\n## 2. 加载并准备所有试验通用的数据... ##")
    current_dir = Path(__file__).resolve().parent
    base_data_path = current_dir.parent.parent / "data" / "SorotokiData" / "MotionData8" / "FilteredDataPos"
    train_path = base_data_path / "80minTrain"
    test_path = base_data_path / "50secTest"
    control_var_name = 'input'
    state_var_name = 'state'
    train_files = sorted(list(train_path.glob('*.mat')))
    test_files = sorted(list(test_path.glob('*.mat')))

    # 仅使用训练数据计算归一化参数
    print("正在计算归一化参数...")
    state_for_norm = np.concatenate([loadmat(f)[state_var_name] for f in train_files], axis=1)
    _, params_state = normalize_data(state_for_norm)
    base_params['params_state'] = params_state
    print("归一化参数计算完毕。")

    # 加载并归一化所有轨迹数据，但不生成窗口序列
    def load_and_normalize(file_list, params_state):
        data_list = []
        for file_path in file_list:
            data = loadmat(file_path)
            state_data_norm, _ = normalize_data(data[state_var_name], params_state)
            control_data = data[control_var_name]  # 控制数据不归一化
            data_list.append({'control': control_data, 'state': state_data_norm})
        return data_list

    normalized_train_data = load_and_normalize(train_files, params_state)
    normalized_test_data = load_and_normalize(test_files, params_state)

    data_dict = {
        'normalized_train_data': normalized_train_data,
        'normalized_test_data': normalized_test_data
    }
    print("数据预加载和归一化完成。")

    # --- 3. 配置网格搜索 (使用参考文件中的参数值) ---
    print("\n## 3. 配置并开始网格搜索... ##")
    ray.init(num_cpus=os.cpu_count(), num_gpus=torch.cuda.device_count(), ignore_reinit_error=True)

    # 核心：使用与参考文件完全相同的参数值进行搜索
    search_space = {
        "delay_step": tune.grid_search([2, 4, 6, 8, 10]),
        "i_multiplier": tune.grid_search([12, 14, 16, 18, 20, 22, 24, 26, 28, 30]),
        "seed": tune.grid_search([42, 3407, 103, 1113, 666, 114514, 2025, 7, 2002, 542])
    }

    # 配置每个试验的资源分配 (与参考文件一致)
    trainable_with_resources = tune.with_resources(train_trial, resources={"cpu": 1, "gpu": 0.2})

    tuner = tune.Tuner(
        tune.with_parameters(trainable_with_resources, data_dict=data_dict, base_params=base_params),
        param_space=search_space,
        run_config=tune.RunConfig(
            name="Hybrid_LTV_GridSearch_Delay_IMult_Seed",
            storage_path=str(BASE_RESULTS_PATH),
            verbose=1
        ),
        tune_config=tune.TuneConfig(
            metric="rmse",
            mode="min"
        )
    )

    results = tuner.fit()

    print(
        f"\n\n{'=' * 80}\n所有网格搜索试验完成! 总耗时: {(time.time() - start_time_total) / 3600:.2f} 小时\n{'=' * 80}")

    # --- 4. 结果分析与聚合 (与参考文件逻辑一致) ---
    if results.errors:
        print("警告: 部分试验出现错误。")

    if results.get_dataframe().empty:
        print("警告: Ray Tune 没有返回任何结果。")
        ray.shutdown()
        return

    results_df = results.get_dataframe()
    full_results_csv_path = BASE_RESULTS_PATH / "grid_search_full_raw_results.csv"
    results_df.to_csv(full_results_csv_path, index=False)
    print(f"所有试验的原始结果（含错误）已保存至: {full_results_csv_path}")

    # 筛选出所有成功报告了有效（非NaN，非无穷大）rmse的试验
    successful_results_df = results_df.dropna(subset=['rmse']).copy()
    successful_results_df = successful_results_df[np.isfinite(successful_results_df['rmse'])]

    if successful_results_df.empty:
        print("所有试验均未能报告有效的RMSE值，无法进行分析。")
        ray.shutdown()
        return

    # 将config字典中的参数展开为单独的列
    search_space_keys = [key for key in search_space.keys() if key != 'seed']
    for param in search_space_keys:
        successful_results_df[param] = successful_results_df['config'].apply(lambda c: c.get(param))

    # 按超参数组合（除了seed）分组，计算RMSE的均值和标准差
    agg_results = successful_results_df.groupby(search_space_keys)['rmse'].agg(
        ['mean', 'std']
    ).reset_index()
    agg_results.rename(columns={'mean': 'avg_rmse', 'std': 'std_rmse'}, inplace=True)
    agg_results.sort_values(by='avg_rmse', inplace=True)

    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 400)

    print("\n\n--- 网格搜索聚合结果 (按平均RMSE排序) ---")
    print(agg_results.to_string(index=False))

    results_csv_path = BASE_RESULTS_PATH / "grid_search_aggregated_results.csv"
    agg_results.to_csv(results_csv_path, index=False)
    print(f"\n聚合结果已保存至: {results_csv_path}")

    # --- 5. 保存最优模型 (与参考文件逻辑一致) ---
    print("\n--- 正在保存最优模型... ---")
    try:
        best_params_combo = agg_results.iloc[0]

        # 筛选出具有最优参数组合的所有试验（不同seed）
        best_trials_df = successful_results_df.copy()
        for param in search_space_keys:
            best_trials_df = best_trials_df[best_trials_df[param] == best_params_combo[param]]

        # 在这些试验中，根据实际的单次rmse值找到最佳的那个试验
        best_trial_row = best_trials_df.loc[best_trials_df['rmse'].idxmin()]
        best_trial_path = Path(best_trial_row['logdir'])
        source_model_path = best_trial_path / "checkpoint" / "final_model.pth"

        if source_model_path.exists():
            # 保存模型
            target_model_path = BASE_RESULTS_PATH / "best_model.pth"
            shutil.copy(source_model_path, target_model_path)
            print(f"✅ 最优模型已成功保存至: {target_model_path}")

            # 保存最优参数为JSON
            best_params_dict = best_trial_row['config']
            target_params_path = BASE_RESULTS_PATH / "best_params.json"
            with open(target_params_path, 'w') as f:
                json.dump(best_params_dict, f, indent=4)
            print(f"✅ 最优参数已成功保存至: {target_params_path}")

            print("\n--- 最优参数组合 ---")
            for key in search_space_keys:
                print(f"{key.replace('_', ' ').title()}: {best_params_combo[key]}")
            print(f"平均 RMSE (跨种子): {best_params_combo['avg_rmse']:.6f}")
            print(
                f"RMSE 标准差: {best_params_combo['std_rmse']:.6f if not pd.isna(best_params_combo['std_rmse']) else 'N/A'}")
            print(f"来源试验的最佳Seed: {best_trial_row['config'].get('seed', 'N/A')}")
            print(f"该试验的RMSE: {best_trial_row['rmse']:.6f}")
        else:
            print(f"❌ 错误: 在最优试验的检查点中未找到 'final_model.pth' 文件。路径: {source_model_path}")

    except (IndexError, KeyError) as e:
        print(f"❌ 保存最优模型时发生错误，可能是因为没有有效的聚合结果: {e}")
    except Exception as e:
        print(f"❌ 保存最优模型时发生未知错误: {e}")

    ray.shutdown()


if __name__ == '__main__':
    main()