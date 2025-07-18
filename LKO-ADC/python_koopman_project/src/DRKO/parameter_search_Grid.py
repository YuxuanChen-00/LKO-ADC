import torch
import numpy as np
from scipy.io import loadmat
import matplotlib
import random
from pathlib import Path
import time
import os
import shutil  # 导入用于文件操作的库
import ray
from ray import tune
from ray.tune import Checkpoint
import pandas as pd

# --- 假设这些自定义函数和类位于您的项目目录中 ---
from evaluate_lstm_lko import evaluate_lstm_lko
from generate_lstm_data import generate_lstm_data
from src.normalize_data import normalize_data
from train_lstm_lko import train_lstm_lko

# --- 全局设置 ---
matplotlib.use('Agg')
BASE_MODEL_SAVE_PATH = Path(__file__).resolve().parent / "models" / "LKO_GridSearch_ray_Final_motion8"


# -----------------------------------------------------------------------------
# 步骤1: 定义训练函数 (无需改动)
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

    set_seed(config['seed'])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    params = base_params.copy()
    params.update(config)
    params['device'] = device
    params['seed'] = config['seed']

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

    net = train_lstm_lko(params, train_data_full, test_data_full)
    net.to(device)
    net.eval()

    final_rmse_scores = []
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
            final_rmse_scores.append(rmse_score)

    mean_rmse_for_this_seed = np.mean(final_rmse_scores)

    checkpoint = None
    if net is not None:
        # 在每个试验的临时工作目录中创建检查点
        # Ray Tune会自动管理这个检查点
        checkpoint_dir = os.path.join(os.getcwd(), "checkpoint")
        os.makedirs(checkpoint_dir, exist_ok=True)
        model_path = os.path.join(checkpoint_dir, "final_model.pth")
        torch.save(net.state_dict(), model_path)
        checkpoint = Checkpoint.from_directory(checkpoint_dir)

    tune.report({"rmse": mean_rmse_for_this_seed}, checkpoint=checkpoint)


# -----------------------------------------------------------------------------
# 步骤2: 主执行流程 (增加保存最优模型的功能)
# -----------------------------------------------------------------------------
def main():
    start_time_total = time.time()
    print("## 1. 设置基础参数和路径... ##")
    BASE_MODEL_SAVE_PATH.mkdir(parents=True, exist_ok=True)

    base_params = {
        'initialLearnRate': 0.005,
        'hidden_size_lstm': 256,
        'hidden_size_mlp': 64,
        'is_norm': True, 'state_size': 6, 'control_size': 6,
        'minLearnRate': 1e-6, 'num_epochs': 100,
        'L1': 1.0, 'L2': 1.0, 'L3': 0.0001,
        'batchSize': 1024, 'patience': 20,
        'lrReduceFactor': 0.2, 'pred_step': 5,
    }

    print("\n## 2. 加载和预处理数据... ##")
    current_dir = Path(__file__).resolve().parent
    base_data_path = current_dir.parent.parent / "data" / "SorotokiData" / "MotionData8" / "FilteredDataPos"
    train_path = base_data_path / "80minTrain"
    test_path = base_data_path / "50secTest"
    control_var_name = 'input'
    state_var_name = 'state'
    train_files = sorted(list(train_path.glob('*.mat')))
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


    train_data_full = {
        'control': np.concatenate(control_train_list, axis=0),
        'state': np.concatenate(state_train_list, axis=0),
        'label': np.concatenate(label_train_list, axis=0)
    }

    test_data_full = []
    for raw_data in raw_test_data:
        state_test_raw, control_test_raw = raw_data['state'], raw_data['control']
        state_test_norm, _ = normalize_data(state_test_raw, params_state)
        control_test_norm, _ = normalize_data(control_test_raw, params_control)
        ctrl_td, state_td, label_td = generate_lstm_data(control_test_norm, state_test_norm, MAX_DELAY,
                                                         base_params['pred_step'])
        test_data_full.append({
            'control': torch.from_numpy(ctrl_td.copy()).float(),
            'state': torch.from_numpy(state_td.copy()).float(),
            'label': torch.from_numpy(label_td.copy()).float()
        })

    print("数据准备完毕。")
    data_dict = {'train_data_full': train_data_full, 'test_data_full': test_data_full}

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

    # --- ✅✅✅ 新增功能：保存最优模型 ✅✅✅ ---
    print("\n--- 正在保存最优模型... ---")
    try:
        # 获取最优试验的结果对象
        # 我们基于聚合后的avg_rmse来找到最优的参数组合，然后再从原始df中筛选出这些试验
        best_params_combo = agg_results.iloc[0]
        best_delay_step = best_params_combo['delay_step']
        best_i_multiplier = best_params_combo['i_multiplier']

        # 从所有成功的试验中，筛选出具有最优(delay_step, i_multiplier)组合的那些试验
        best_trials_df = results_df_successful[
            (results_df_successful['delay_step'] == best_delay_step) &
            (results_df_successful['i_multiplier'] == best_i_multiplier)
            ]

        # 在这些最优组合的试验中，再根据单个的rmse值找到最好的那一个
        best_trial_row = best_trials_df.loc[best_trials_df['rmse'].idxmin()]

        # 获取最优试验的路径
        best_trial_path = Path(best_trial_row['logdir'])

        # 寻找检查点文件夹
        checkpoint_dirs = list(best_trial_path.glob("checkpoint_*"))
        if checkpoint_dirs:
            # 找到最新的检查点（通常只有一个）
            best_checkpoint_dir = max(checkpoint_dirs, key=os.path.getmtime)
            source_model_path = best_checkpoint_dir / "final_model.pth"

            if source_model_path.exists():
                target_model_path = BASE_MODEL_SAVE_PATH / "best_model.pth"
                # 复制文件
                shutil.copy(source_model_path, target_model_path)
                print(f"✅ 最优模型已成功保存至: {target_model_path}")
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