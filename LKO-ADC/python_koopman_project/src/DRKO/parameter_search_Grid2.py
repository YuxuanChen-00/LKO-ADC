import torch
import numpy as np
from scipy.io import loadmat
import matplotlib
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

# --- 假设这些自定义函数和类位于您的项目目录中 ---
# 请确保这些import路径是正确的
from evaluate_lstm_lko import evaluate_lstm_lko, evaluate_lstm_lko2
from generate_lstm_data import generate_lstm_data
from src.normalize_data import normalize_data
from train_lstm_lko import train_lstm_lko
from src.DRKO.lko_lstm_network import LKO_lstm_Network  # 假设网络定义在这里

# --- 全局设置 ---
matplotlib.use('Agg')
BASE_MODEL_SAVE_PATH = Path(__file__).resolve().parent / "models" / "LKO_GridSearch_ray_Final_motion8_update"


# -----------------------------------------------------------------------------
# 步骤1: 定义训练函数 (已修改)
# -----------------------------------------------------------------------------
def train_model(config, data_dict, base_params):
    """
    Ray Tune的单次试验执行函数。
    此函数现在负责为自己的配置参数独立生成数据。
    """

    # --- 种子设置 ---
    def set_seed(seed):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    set_seed(config['seed'])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- 参数配置 ---
    params = base_params.copy()
    params.update(config)
    params['device'] = device
    params['seed'] = config['seed']
    params['hidden_size_lstm'] = int(params['hidden_size_lstm'])
    params['hidden_size_mlp'] = int(params['hidden_size_mlp'])
    params['PhiDimensions'] = params['delay_step'] * params['i_multiplier']
    params['output_size'] = params['PhiDimensions']

    # --- 新增逻辑：在试验内部生成专属数据 ---
    print(f"Trial (seed={params['seed']}, delay_step={params['delay_step']}): Generating specific dataset...")

    # 从 data_dict 中解包归一化后的原始数据
    normalized_train_data = data_dict['normalized_train_data']
    normalized_test_data = data_dict['normalized_test_data']

    # 1. 为当前 delay_step 生成训练数据
    control_train_list, state_train_list, label_train_list = [], [], []
    for raw_data in normalized_train_data:
        # 使用当前试验的 params['delay_step']
        ctrl_td, state_td, label_td = generate_lstm_data(
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

    # 2. 为当前 delay_step 生成测试数据
    test_data_full_cpu = []
    for raw_data in normalized_test_data:
        # 使用当前试验的 params['delay_step']
        ctrl_td, state_td, label_td = generate_lstm_data(
            raw_data['control'], raw_data['state'], params['delay_step'], params['pred_step']
        )
        test_data_full_cpu.append({
            'control': torch.from_numpy(ctrl_td.copy()).float(),
            'state': torch.from_numpy(state_td.copy()).float(),
            'label': torch.from_numpy(label_td.copy()).float()
        })

    # --- 训练与评估 ---
    test_data_full = [
        {'control': d['control'].to(device), 'state': d['state'].to(device), 'label': d['label'].to(device)}
        for d in test_data_full_cpu
    ]

    net = train_lstm_lko(params, train_data_full, test_data_full)
    net.to(device)
    net.eval()

    final_rmse_scores = []
    with torch.no_grad():
        for test_set in test_data_full:
            start_index = 10 - params['delay_step']
            initial_state_sequence = test_set['state'][start_index, :, :]

            rmse_score, _, _ = evaluate_lstm_lko2(
                net,
                test_set['control'][start_index:],
                initial_state_sequence,
                test_set['label'][start_index:],
                params['params_state'],
                base_params['is_norm']
            )
            final_rmse_scores.append(rmse_score)

    mean_rmse_for_this_seed = np.mean(final_rmse_scores)

    # --- 保存检查点 ---
    checkpoint = None
    if net is not None:
        checkpoint_dir = os.path.join(os.getcwd(), "checkpoint")
        os.makedirs(checkpoint_dir, exist_ok=True)
        model_path = os.path.join(checkpoint_dir, "final_model.pth")
        torch.save(net.state_dict(), model_path)
        checkpoint = Checkpoint.from_directory(checkpoint_dir)

    tune.report({"rmse": mean_rmse_for_this_seed}, checkpoint=checkpoint)


# -----------------------------------------------------------------------------
# 步骤2: 主执行流程 (已修改)
# -----------------------------------------------------------------------------
def main():
    start_time_total = time.time()
    print("## 1. 设置基础参数和路径... ##")
    BASE_MODEL_SAVE_PATH.mkdir(parents=True, exist_ok=True)

    base_params = {
        'initialLearnRate': 0.005, 'hidden_size_lstm': 256, 'hidden_size_mlp': 64,
        'is_norm': True, 'state_size': 6, 'control_size': 6, 'minLearnRate': 1e-6,
        'num_epochs': 100, 'L1': 1.0, 'L2': 1.0, 'L3': 0.0001, 'batchSize': 1024,
        'patience': 20, 'lrReduceFactor': 0.2, 'pred_step': 5,
    }

    # --- 修改数据准备逻辑 ---
    print("\n## 2. 加载和预处理数据... ##")
    current_dir = Path(__file__).resolve().parent
    base_data_path = current_dir.parent.parent / "data" / "SorotokiData" / "MotionData8" / "FilteredDataPos"
    train_path = base_data_path / "80minTrain"
    test_path = base_data_path / "50secTest"
    control_var_name = 'input'
    state_var_name = 'state'
    train_files = sorted(list(train_path.glob('*.mat')))
    test_files = sorted(list(test_path.glob('*.mat')))

    # --- 归一化参数计算 ---
    print("正在计算归一化参数...")
    state_for_norm = np.array([]).reshape(base_params['state_size'], 0)
    control_for_norm = np.array([]).reshape(base_params['control_size'], 0)
    for file_path in train_files:
        data = loadmat(file_path)
        state_for_norm = np.concatenate((state_for_norm, data[state_var_name]), axis=1)
        control_for_norm = np.concatenate((control_for_norm, data[control_var_name]), axis=1)

    _, params_state = normalize_data(state_for_norm)
    _, params_control = normalize_data(control_for_norm)
    base_params['params_state'] = params_state
    base_params['params_control'] = params_control
    print("归一化参数计算完毕。")

    # --- 新逻辑：准备归一化后的原始数据列表 ---
    print("正在准备将分发给每个试验的、归一化后的原始数据...")

    # 1. 加载并归一化训练数据
    normalized_train_data = []
    for file_path in train_files:
        data = loadmat(file_path)
        state_data_norm, _ = normalize_data(data[state_var_name], params_state)
        control_data_norm, _ = normalize_data(data[control_var_name], params_control)
        normalized_train_data.append({'control': control_data_norm, 'state': state_data_norm})

    # 2. 加载并归一化测试数据
    normalized_test_data = []
    for file_path in test_files:
        data = loadmat(file_path)
        state_test_norm, _ = normalize_data(data[state_var_name], params_state)
        control_test_norm, _ = normalize_data(data[control_var_name], params_control)
        normalized_test_data.append({'control': control_test_norm, 'state': state_test_norm})

    # 这个 data_dict 将被传递给每个独立的试验
    data_dict = {
        'normalized_train_data': normalized_train_data,
        'normalized_test_data': normalized_test_data
    }
    print("数据准备完毕。")

    # --- 网格搜索配置 ---
    print("\n## 3. 配置并开始网格搜索... ##")
    ray.init(num_cpus=64, num_gpus=torch.cuda.device_count(), ignore_reinit_error=True)

    search_space = {
        "delay_step": tune.grid_search([2, 4, 6, 8, 10]),
        "i_multiplier": tune.grid_search([12, 14, 16, 18, 20, 22, 24, 26, 28, 30]),
        "seed": tune.grid_search([42, 3407, 103, 1113, 666, 114514, 2025, 7, 2002, 542])
    }

    trainable_with_resources = tune.with_resources(train_model, resources={"cpu": 1, "gpu": 0.1})

    tuner = tune.Tuner(
        tune.with_parameters(trainable_with_resources, data_dict=data_dict, base_params=base_params),
        param_space=search_space,
        run_config=tune.RunConfig(
            name="GridSearch_Delay_IMult_Seed",
            storage_path=str(BASE_MODEL_SAVE_PATH),
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

    # --- 结果分析与聚合 ---
    results_df = results.get_dataframe()

    if results_df.empty:
        print("没有可供分析的有效试验结果（结果集为空）。")
        ray.shutdown()
        return

    results_df_successful = results_df[results_df.error.isna()].copy()
    if results_df_successful.empty:
        print("所有试验均执行失败，没有可供分析的有效结果。")
        ray.shutdown()
        return

    for param in search_space.keys():
        results_df_successful[param] = results_df_successful['config'].apply(lambda c: c.get(param))

    agg_results = results_df_successful.groupby(['delay_step', 'i_multiplier'])['rmse'].agg(
        ['mean', 'std']).reset_index()
    agg_results.rename(columns={'mean': 'avg_rmse', 'std': 'std_rmse'}, inplace=True)
    agg_results.sort_values(by='avg_rmse', inplace=True)

    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 400)

    print("\n\n--- 网格搜索聚合结果 (按平均RMSE排序) ---")
    print(agg_results.to_string(index=False))

    results_csv_path = BASE_MODEL_SAVE_PATH / "grid_search_aggregated_results.csv"
    agg_results.to_csv(results_csv_path, index=False)
    print(f"\n聚合结果已保存至: {results_csv_path}")

    full_results_csv_path = BASE_MODEL_SAVE_PATH / "grid_search_full_raw_results.csv"
    results_df.to_csv(full_results_csv_path, index=False)
    print(f"所有试验的原始结果（含错误）已保存至: {full_results_csv_path}")

    # --- 保存最优模型 ---
    print("\n--- 正在保存最优模型... ---")
    try:
        best_params_combo = agg_results.iloc[0]
        best_delay_step = best_params_combo['delay_step']
        best_i_multiplier = best_params_combo['i_multiplier']

        best_trials_df = results_df_successful[
            (results_df_successful['delay_step'] == best_delay_step) &
            (results_df_successful['i_multiplier'] == best_i_multiplier)
            ]

        best_trial_row = best_trials_df.loc[best_trials_df['rmse'].idxmin()]
        best_trial_path = Path(best_trial_row['logdir'])
        checkpoint_dirs = list(best_trial_path.glob("checkpoint_*"))

        if checkpoint_dirs:
            best_checkpoint_dir = max(checkpoint_dirs, key=os.path.getmtime)
            source_model_path = best_checkpoint_dir / "final_model.pth"

            if source_model_path.exists():
                # 保存模型
                target_model_path = BASE_MODEL_SAVE_PATH / "best_model.pth"
                shutil.copy(source_model_path, target_model_path)
                print(f"✅ 最优模型已成功保存至: {target_model_path}")

                # 保存最优参数为JSON
                best_params_dict = best_trial_row['config']
                target_params_path = BASE_MODEL_SAVE_PATH / "best_params.json"
                with open(target_params_path, 'w') as f:
                    json.dump(best_params_dict, f, indent=4)
                print(f"✅ 最优参数已成功保存至: {target_params_path}")

                print("\n--- 最优参数组合 ---")
                print(f"Delay Step: {best_delay_step}")
                print(f"i Multiplier: {best_i_multiplier}")
                print(f"平均 RMSE (跨种子): {best_params_combo['avg_rmse']:.6f}")
                print(f"RMSE 标准差: {best_params_combo['std_rmse']:.6f}")
                print(f"来源试验 (Seed): {best_trial_row['seed']}")
                print(f"来源试验RMSE: {best_trial_row['rmse']:.6f}")
            else:
                print(f"❌ 错误: 在最优试验的检查点中未找到 'final_model.pth' 文件。路径: {source_model_path}")
        else:
            print(f"❌ 错误: 在最优试验文件夹中未找到任何检查点目录。路径: {best_trial_path}")

    except Exception as e:
        print(f"❌ 保存最优模型时发生错误: {e}")

    ray.shutdown()


if __name__ == '__main__':
    main()