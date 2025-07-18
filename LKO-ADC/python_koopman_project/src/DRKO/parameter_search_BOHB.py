import torch
import numpy as np
from scipy.io import loadmat
import matplotlib
import random
from pathlib import Path
import time
import os
import ray
from ray import tune
from ray.tune.schedulers import HyperBandForBOHB
from ray.tune.search.bohb import TuneBOHB
import ConfigSpace as CS
import pandas as pd
# --- 核心修正：恢复到之前可以正常工作的导入方式 ---
from ray.train import Checkpoint

# --- 假设这些自定义函数和类位于您的项目目录中 ---
from evaluate_lstm_lko import evaluate_lstm_lko
from generate_lstm_data import generate_lstm_data
from src.normalize_data import normalize_data
from train_lstm_lko import train_lstm_lko

# --- 全局设置 ---
matplotlib.use('Agg')
BASE_MODEL_SAVE_PATH = Path(__file__).resolve().parent / "models" / "LKO_Hybrid_Search_ray_OnTheFly"


# -----------------------------------------------------------------------------
# 步骤1: 定义训练函数 (已重构，在函数内部生成数据)
# -----------------------------------------------------------------------------
def train_model(config, raw_data_dict, base_params):
    def set_seed(seed):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    params = base_params.copy()
    params.update(config)  # 将BOHB生成的超参数合并进来

    # 确保整型参数的类型正确
    params['hidden_size_lstm'] = int(params['hidden_size_lstm'])
    params['hidden_size_mlp'] = int(params['hidden_size_mlp'])
    params['num_epochs'] = int(params['num_epochs'])
    params['pred_step'] = int(params['pred_step'])
    params['device'] = device  # 将device添加到params中

    params['PhiDimensions'] = params['delay_step'] * params['i_multiplier']
    params['output_size'] = params['PhiDimensions']

    # =================================================================================
    # --- 新增核心修改: 根据当前试验的 pred_step 和 delay_step 实时生成数据 ---
    # --- 注意: 此方法遵循了您的最新指示，但会显著增加每次试验的计算开销。---
    # =================================================================================
    params_state = params['params_state']
    params_control = params['params_control']
    current_delay_step = params['delay_step']
    current_pred_step = params['pred_step']
    raw_train_data = raw_data_dict['raw_train_data']
    raw_test_data = raw_data_dict['raw_test_data']

    # --- 1. 生成训练数据 ---
    control_train_list, state_train_list, label_train_list = [], [], []
    for raw_data in raw_train_data:
        state_data, control_data = raw_data['state'], raw_data['control']
        state_data_norm, _ = normalize_data(state_data, params_state)
        control_data_norm, _ = normalize_data(control_data, params_control)
        ctrl_td, state_td, label_td = generate_lstm_data(
            control_data_norm, state_data_norm, current_delay_step, current_pred_step
        )
        control_train_list.append(ctrl_td)
        state_train_list.append(state_td)
        label_train_list.append(label_td)

    train_data_full = {
        'control': np.concatenate(control_train_list, axis=0),
        'state': np.concatenate(state_train_list, axis=0),
        'label': np.concatenate(label_train_list, axis=0)
    }

    # --- 2. 生成测试数据 ---
    test_data_full_cpu = []
    for raw_data in raw_test_data:
        state_test_raw, control_test_raw = raw_data['state'], raw_data['control']
        state_test_norm, _ = normalize_data(state_test_raw, params_state)
        control_test_norm, _ = normalize_data(control_test_raw, params_control)
        ctrl_td, state_td, label_td = generate_lstm_data(
            control_test_norm, state_test_norm, current_delay_step, current_pred_step
        )
        test_data_full_cpu.append(
            {'control': torch.from_numpy(ctrl_td).float(),
             'state': torch.from_numpy(state_td).float(),
             'label': torch.from_numpy(label_td).float()}
        )
    # =================================================================================
    # --- 数据实时生成结束 ---
    # =================================================================================

    test_data_full = [
        {'control': d['control'].to(device), 'state': d['state'].to(device), 'label': d['label'].to(device)}
        for d in test_data_full_cpu
    ]

    seed_list = [3407]
    per_seed_scores = {}
    last_trained_net = None

    for seed in seed_list:
        set_seed(seed)
        params['seed'] = seed
        net = train_lstm_lko(params, train_data_full, test_data_full)
        last_trained_net = net

        net.to(device)
        net.eval()

        final_rmse_scores_one_seed = []
        with torch.no_grad():
            for test_set in test_data_full:
                # --- 修正: 移除硬编码的索引'10'，使评估逻辑更健壮 ---
                # 从每个测试文件的第一个可用序列开始评估
                if test_set['state'].shape[0] > 0:
                    initial_state_sequence = test_set['state'][0]
                    rmse_score, _, _ = evaluate_lstm_lko(
                        net,
                        test_set['control'],
                        initial_state_sequence,
                        test_set['label'],
                        params['params_state'],
                        base_params['is_norm']
                    )
                    final_rmse_scores_one_seed.append(rmse_score)

        if final_rmse_scores_one_seed:
            mean_rmse_one_seed = np.mean(final_rmse_scores_one_seed)
        else:
            mean_rmse_one_seed = float('inf')  # 如果没有有效的测试数据点，则返回一个极大值

        per_seed_scores[seed] = mean_rmse_one_seed

    score_values = list(per_seed_scores.values())
    avg_rmse = np.mean(score_values)
    std_rmse = np.std(score_values)

    checkpoint = None
    if last_trained_net is not None:
        checkpoint_dir = os.path.join(os.getcwd(), "checkpoint")
        os.makedirs(checkpoint_dir, exist_ok=True)
        model_path = os.path.join(checkpoint_dir, "final_model.pth")
        torch.save(last_trained_net.state_dict(), model_path)
        checkpoint = Checkpoint.from_directory(checkpoint_dir)

    metrics_to_report = {
        "avg_rmse": avg_rmse,
        "std_rmse": std_rmse,
    }
    for seed, score in per_seed_scores.items():
        metrics_to_report[f"seed_{seed}"] = score

    tune.report(metrics_to_report, checkpoint=checkpoint)


# -----------------------------------------------------------------------------
# 步骤2: 主执行流程 (已修改，仅加载原始数据)
# -----------------------------------------------------------------------------
def main():
    start_time_total = time.time()
    print("## 1. 设置基础参数和路径... ##")
    BASE_MODEL_SAVE_PATH.mkdir(parents=True, exist_ok=True)

    base_params = {
        'is_norm': True, 'state_size': 6, 'control_size': 6,
        'batchSize': 1024, 'patience': 20,
        'lrReduceFactor': 0.2,
        'L2': 1.0, 'L3': 1.0,
    }

    print("\n## 2. 加载原始数据和计算标准化参数... ##")
    current_dir = Path(__file__).resolve().parent
    base_data_path = current_dir.parent.parent / "data" / "SorotokiData" / "MotionData8" / "FilteredDataPos"
    train_path = base_data_path / "80minTrain"
    test_path = base_data_path / "50secTest"
    control_var_name = 'input'
    state_var_name = 'state'
    train_files = sorted(list(train_path.glob('*.mat')));
    test_files = sorted(list(test_path.glob('*.mat')))

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

    # --- 修改: 只加载原始数据，不在此处生成序列 ---
    raw_train_data = [{'control': loadmat(f)[control_var_name], 'state': loadmat(f)[state_var_name]} for f in
                      train_files]
    raw_test_data = [{'control': loadmat(f)[control_var_name], 'state': loadmat(f)[state_var_name]} for f in test_files]
    print("原始数据加载完毕。序列化数据将在每次试验中动态生成。")
    raw_data_dict = {'raw_train_data': raw_train_data, 'raw_test_data': raw_test_data}
    # --- 修改结束 ---

    print("\n## 3. 配置并开始混合超参数搜索... ##")
    ray.init(num_cpus=os.cpu_count() or 1, num_gpus=torch.cuda.device_count(), ignore_reinit_error=True)

    delay_steps_to_search = [6]
    i_multipliers_to_search = [20]
    print(f"网格搜索范围: delay_step = {delay_steps_to_search}, i_multiplier = {i_multipliers_to_search}")

    bohb_config_space = CS.ConfigurationSpace()
    bohb_config_space.add(CS.CategoricalHyperparameter("hidden_size_lstm", [32, 64, 128, 256, 512]))
    bohb_config_space.add(CS.CategoricalHyperparameter("hidden_size_mlp", [16, 32, 64, 128, 256]))
    bohb_config_space.add(CS.UniformFloatHyperparameter("initialLearnRate", lower=1e-4, upper=1e-2, log=True))
    bohb_config_space.add(CS.UniformIntegerHyperparameter("num_epochs", lower=80, upper=200))
    # --- 注意: pred_step 的搜索范围，每次试验都会重新生成数据 ---
    bohb_config_space.add(CS.UniformIntegerHyperparameter("pred_step", lower=1, upper=10))
    bohb_config_space.add(CS.UniformFloatHyperparameter("L1", lower=1e-3, upper=1e3, log=True))
    bohb_config_space.add(CS.UniformFloatHyperparameter("minLearnRate", lower=1e-8, upper=1e-5, log=True))

    all_results_collector = []
    for delay in delay_steps_to_search:
        for i_mult in i_multipliers_to_search:
            grid_start_time = time.time()
            print(f"\n{'=' * 30}\n开始搜索: delay_step = {delay}, i_multiplier = {i_mult}\n{'=' * 30}")

            current_base_params = base_params.copy()
            current_base_params['delay_step'] = delay
            current_base_params['i_multiplier'] = i_mult

            bohb_search = TuneBOHB(space=bohb_config_space, metric="avg_rmse", mode="min", seed=123)

            max_epochs_in_search = bohb_config_space.get_hyperparameter("num_epochs").upper
            bohb_scheduler = HyperBandForBOHB(time_attr="training_iteration", max_t=max_epochs_in_search,
                                              reduction_factor=3, stop_last_trials=False)

            trainable_with_resources = tune.with_resources(train_model, resources={"cpu": 1, "gpu": 0.1})

            # --- 修改: 传递 raw_data_dict 而不是预处理好的 data_dict ---
            tuner = tune.Tuner(
                tune.with_parameters(trainable_with_resources, raw_data_dict=raw_data_dict,
                                     base_params=current_base_params),
                run_config=tune.RunConfig(name=f"Grid_D{delay}_I{i_mult}", storage_path=str(BASE_MODEL_SAVE_PATH),
                                          verbose=1),
                tune_config=tune.TuneConfig(scheduler=bohb_scheduler, search_alg=bohb_search, num_samples=100,
                                            metric="avg_rmse", mode="min"))
            # --- 修改结束 ---
            results = tuner.fit()

            for result in results:
                if result.metrics and result.path:
                    trial_config = result.config
                    trial_data = {
                        "delay_step": delay,
                        "i_multiplier": i_mult,
                        **trial_config,
                        **result.metrics
                    }
                    all_results_collector.append(trial_data)

            grid_duration = (time.time() - grid_start_time) / 60
            print(f"\n搜索完成: delay_step = {delay}, i_multiplier = {i_mult}, 耗时: {grid_duration:.2f} 分钟")

    print(f"\n\n{'=' * 80}\n所有超参数搜索完成! 总耗时: {(time.time() - start_time_total) / 3600:.2f} 小时\n{'=' * 80}")

    if not all_results_collector:
        print("没有可供分析的有效试验结果。")
        ray.shutdown()
        return

    results_df = pd.DataFrame(all_results_collector)

    print("\n\n--- 所有试验的综合结果 ---")
    print(f"总共完成的试验次数: {len(results_df)}")

    # 结果展示部分保持不变
    if 'delay_step' in results_df.columns and 'i_multiplier' in results_df.columns:
        results_df['PhiDimensions'] = results_df['delay_step'] * results_df['i_multiplier']

    seed_columns = sorted([col for col in results_df.columns if str(col).startswith('seed_')])

    display_columns = [
                          'delay_step', 'i_multiplier', 'PhiDimensions',
                          'hidden_size_lstm', 'hidden_size_mlp', 'initialLearnRate',
                          'num_epochs', 'pred_step', 'L1', 'minLearnRate',
                          'avg_rmse', 'std_rmse'
                      ] + seed_columns

    final_display_columns = [col for col in display_columns if col in results_df.columns]

    rename_map = {
        'delay_step': 'Delay', 'i_multiplier': 'i-Mult', 'PhiDimensions': 'PhiDims',
        'hidden_size_lstm': 'LSTM Size', 'hidden_size_mlp': 'MLP Size', 'initialLearnRate': 'LR',
        'num_epochs': 'Epochs', 'pred_step': 'Pred Step', 'L1': 'L1', 'minLearnRate': 'Min LR',
        'avg_rmse': 'Avg RMSE', 'std_rmse': 'Std RMSE'
    }
    for col in seed_columns:
        seed_value = col.split('_', 1)[1]
        rename_map[col] = seed_value

    detailed_results = results_df[final_display_columns].copy()
    detailed_results.rename(columns=rename_map, inplace=True)
    detailed_results.sort_values(by='Avg RMSE', inplace=True)

    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 400)

    print(detailed_results.to_string(index=False))

    results_csv_path = BASE_MODEL_SAVE_PATH / "all_hybrid_search_results_on_the_fly.csv"
    detailed_results.to_csv(results_csv_path, index=False)
    print(f"\n所有试验结果已保存至: {results_csv_path}")

    if not detailed_results.empty:
        best_result_row = detailed_results.iloc[0]
        print("\n\n--- 最优试验结果 ---")
        print(f"目标值 (Avg RMSE): {best_result_row['Avg RMSE']:.6f}")
        if 'Std RMSE' in best_result_row:
            print(f"Std Dev on RMSE: {best_result_row['Std RMSE']:.6f}")
        print("\n--- 最优超参数 ---")
        print(best_result_row.to_string())
    else:
        print("未找到有效结果以确定最优配置。")

    ray.shutdown()


if __name__ == '__main__':
    main()