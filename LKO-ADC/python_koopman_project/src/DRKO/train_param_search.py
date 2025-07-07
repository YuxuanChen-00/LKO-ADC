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
BASE_MODEL_SAVE_PATH = Path(__file__).resolve().parent / "models" / "LKO_Hybrid_Search_ray_Final"


# -----------------------------------------------------------------------------
# 步骤1: 定义训练函数 (功能正确，无需改动)
# -----------------------------------------------------------------------------
def train_model(config, data_dict, base_params):
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
    params.update(config)
    params['device'] = device

    params['hidden_size_lstm'] = int(params['hidden_size_lstm'])
    params['hidden_size_mlp'] = int(params['hidden_size_mlp'])

    params['PhiDimensions'] = params['delay_step'] * params['i_multiplier']
    params['output_size'] = params['PhiDimensions']

    train_data_full = data_dict['train_data_full']
    test_data_full_cpu = data_dict['test_data_full']

    test_data_full = [
        {'control': d['control'].to(device), 'state': d['state'].to(device), 'label': d['label'].to(device)}
        for d in test_data_full_cpu
    ]

    seed_list = [42, 3407, 103, 1113, 114514]

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
                initial_state_sequence = test_set['state'][10 - params['delay_step'], :, :]
                rmse_score, _, _ = evaluate_lstm_lko(
                    net,
                    test_set['control'][10 - params['delay_step']:],
                    initial_state_sequence,
                    test_set['label'][10 - params['delay_step']:],
                    params['params_state'],
                    base_params['is_norm']
                )
                final_rmse_scores_one_seed.append(rmse_score)
        mean_rmse_one_seed = np.mean(final_rmse_scores_one_seed)
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
        "delay_step": params['delay_step'],
        "i_multiplier": params['i_multiplier'],
    }
    for seed, score in per_seed_scores.items():
        metrics_to_report[f"seed_{seed}"] = score

    tune.report(metrics_to_report, checkpoint=checkpoint)


# -----------------------------------------------------------------------------
# 步骤2: 主执行流程 (功能正确，无需改动)
# -----------------------------------------------------------------------------
def main():
    start_time_total = time.time()
    print("## 1. 设置基础参数和路径... ##")
    BASE_MODEL_SAVE_PATH.mkdir(parents=True, exist_ok=True)
    base_params = {
        'is_norm': True, 'state_size': 6, 'control_size': 6,
        'minLearnRate': 1e-6, 'num_epochs': 200,
        'L1': 1.0, 'L2': 1.0, 'L3': 0.0001,
        'batchSize': 8172, 'patience': 20,
        'lrReduceFactor': 0.2, 'pred_step': 5,
    }

    print("\n## 2. 加载和预处理数据... ##")
    current_dir = Path(__file__).resolve().parent
    base_data_path = current_dir.parent.parent / "data" / "SorotokiData" / "MotionData2" / "FilteredDataPos"
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
    raw_train_data = [{'control': loadmat(f)[control_var_name], 'state': loadmat(f)[state_var_name]} for f in
                      train_files]
    raw_test_data = [{'control': loadmat(f)[control_var_name], 'state': loadmat(f)[state_var_name]} for f in test_files]
    MAX_DELAY = 10
    print(f"为最大delay_step={MAX_DELAY}生成训练/测试数据...")
    control_train_list, state_train_list, label_train_list = [], [], []
    for raw_data in raw_train_data:
        state_data, control_data = raw_data['state'], raw_data['control']
        state_data_norm, _ = normalize_data(state_data, params_state)
        control_data_norm, _ = normalize_data(control_data, params_control)
        ctrl_td, state_td, label_td = generate_lstm_data(control_data_norm, state_data_norm, MAX_DELAY,
                                                         base_params['pred_step'])
        control_train_list.append(ctrl_td)
        state_train_list.append(state_td)
        label_train_list.append(label_td)
    train_data_full = {'control': np.concatenate(control_train_list, axis=0),
                       'state': np.concatenate(state_train_list, axis=0),
                       'label': np.concatenate(label_train_list, axis=0)}
    test_data_full = []
    for raw_data in raw_test_data:
        state_test_raw, control_test_raw = raw_data['state'], raw_data['control']
        state_test_norm, _ = normalize_data(state_test_raw, params_state)
        control_test_norm, _ = normalize_data(control_test_raw, params_control)
        ctrl_td, state_td, label_td = generate_lstm_data(control_test_norm, state_test_norm, MAX_DELAY,
                                                         base_params['pred_step'])
        test_data_full.append(
            {'control': torch.from_numpy(ctrl_td).float(), 'state': torch.from_numpy(state_td).float(),
             'label': torch.from_numpy(label_td).float()})
    print("数据准备完毕。")
    data_dict = {'train_data_full': train_data_full, 'test_data_full': test_data_full}

    print("\n## 3. 配置并开始混合超参数搜索... ##")
    ray.init(num_cpus=os.cpu_count() or 1, num_gpus=torch.cuda.device_count(), ignore_reinit_error=True)

    delay_steps_to_search = [2, 4, 6, 8, 10]
    i_multipliers_to_search = [12, 14, 16, 18, 20, 22, 24, 26, 28, 30]
    print(f"网格搜索范围: delay_step = {delay_steps_to_search}, i_multiplier = {i_multipliers_to_search}")

    bohb_config_space = CS.ConfigurationSpace()
    bohb_config_space.add(CS.CategoricalHyperparameter("hidden_size_lstm", [64, 128, 256]))
    bohb_config_space.add(CS.CategoricalHyperparameter("hidden_size_mlp", [64, 128, 256]))
    bohb_config_space.add(CS.UniformFloatHyperparameter("initialLearnRate", lower=1e-4, upper=1e-2, log=True))

    all_results_collector = []
    for delay in delay_steps_to_search:
        for i_mult in i_multipliers_to_search:
            grid_start_time = time.time()
            print(f"\n{'=' * 30}\n开始搜索: delay_step = {delay}, i_multiplier = {i_mult}\n{'=' * 30}")
            current_base_params = base_params.copy()
            current_base_params['delay_step'] = delay
            current_base_params['i_multiplier'] = i_mult
            bohb_search = TuneBOHB(space=bohb_config_space, metric="avg_rmse", mode="min", seed=123)
            bohb_scheduler = HyperBandForBOHB(time_attr="training_iteration", max_t=base_params['num_epochs'],
                                              reduction_factor=3, stop_last_trials=False)
            trainable_with_resources = tune.with_resources(train_model, resources={"cpu": 1, "gpu": 0.25})
            tuner = tune.Tuner(
                tune.with_parameters(trainable_with_resources, data_dict=data_dict, base_params=current_base_params),
                run_config=tune.RunConfig(name=f"Grid_D{delay}_I{i_mult}", storage_path=str(BASE_MODEL_SAVE_PATH),
                                          verbose=1),
                tune_config=tune.TuneConfig(scheduler=bohb_scheduler, search_alg=bohb_search, num_samples=10,
                                            metric="avg_rmse", mode="min"))
            results = tuner.fit()
            for result in results:
                if result.metrics:
                    trial_data = result.config
                    trial_data.update(result.metrics)
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

    results_df['PhiDimensions'] = results_df['delay_step'] * results_df['i_multiplier']
    seed_columns = sorted([col for col in results_df.columns if str(col).startswith('seed_')])
    display_columns = [
                          'delay_step', 'i_multiplier', 'PhiDimensions',
                          'hidden_size_lstm', 'hidden_size_mlp', 'initialLearnRate',
                          'avg_rmse', 'std_rmse'
                      ] + seed_columns
    final_display_columns = [col for col in display_columns if col in results_df.columns]

    rename_map = {
        'delay_step': 'Delay Step', 'i_multiplier': 'i Multiplier', 'PhiDimensions': 'Phi Dims',
        'hidden_size_lstm': 'LSTM Size', 'hidden_size_mlp': 'MLP Size', 'initialLearnRate': 'LR',
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

    results_csv_path = BASE_MODEL_SAVE_PATH / "all_hybrid_search_results.csv"
    detailed_results.to_csv(results_csv_path, index=False)
    print(f"\n所有试验结果已保存至: {results_csv_path}")

    if not detailed_results.empty:
        best_result_row = detailed_results.iloc[0]
        print("\n\n--- 最优试验结果 ---")
        print(f"目标值 (Avg RMSE): {best_result_row['Avg RMSE']:.6f}")
        print(f"Std Dev on RMSE: {best_result_row['Std RMSE']:.6f}")
        print("\n--- 最优超参数 ---")
        print(best_result_row.to_string())
    else:
        print("未找到有效结果以确定最优配置。")

    ray.shutdown()


if __name__ == '__main__':
    main()
