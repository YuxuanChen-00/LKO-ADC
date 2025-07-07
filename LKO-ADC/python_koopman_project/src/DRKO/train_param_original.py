# 文件名: main_hyperparameter_search.py

import torch
import numpy as np
from scipy.io import loadmat
import matplotlib
import random  # <--- ADDED: 导入 random 库

# 使用非交互式后端，避免在服务器上弹出图形窗口
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
import time

# --- 假设这些自定义函数和类位于您的项目目录中 ---
from evaluate_lstm_lko import evaluate_lstm_lko
from generate_lstm_data import generate_lstm_data
from src.normalize_data import normalize_data
from train_lstm_lko import train_lstm_lko


# <--- ADDED: 全局随机种子设置函数 ---
def set_seed(seed):
    """
    为所有相关库设置随机种子以确保实验的可复现性。
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # 为所有GPU设置种子
    # 确保CUDA操作的确定性, 这对于完全的可复现性至关重要
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def run_experiment(params, train_data, test_data, is_norm, params_state, model_save_path):
    """
    运行一次完整的训练和评估流程 (针对单个随机种子)。
    """
    # <--- MODIFIED: 从 params 中获取种子，并在打印信息中包含它 ---
    seed = params['seed']
    print(f"\n--- 开始训练: delay={params['delay_step']}, phi={params['PhiDimensions']}, seed={seed} ---")

    # ==========================================================================
    # 4. 训练网络
    # ==========================================================================
    net = train_lstm_lko(params, train_data, test_data)

    # <--- MODIFIED: 在模型文件名中添加种子信息，避免覆盖 ---
    model_filename = f"model_delay_{params['delay_step']}_phi_{params['PhiDimensions']}_seed_{seed}.pth"
    final_model_path = model_save_path / model_filename
    torch.save(net.state_dict(), final_model_path)
    print(f"训练完成。模型已保存至: '{final_model_path}'")

    # ==========================================================================
    # 5. 最终评估和绘图
    # ==========================================================================
    print(f"\n## 正在为 seed={seed} 进行最终评估和绘图... ##")
    final_rmse_scores = []
    net.eval()
    device = params['device']
    net.to(device)

    for i, test_set in enumerate(test_data):
        delay_step = params['delay_step']
        control_test = test_set['control']
        state_test = test_set['state']
        label_test = test_set['label']
        initial_state_sequence = state_test[10 - delay_step, :, :]
        with torch.no_grad():
            rmse_score, y_true, y_pred = evaluate_lstm_lko(net, control_test[10 - delay_step:],
                                                           initial_state_sequence,
                                                           label_test[10 - delay_step:], params_state, is_norm)
        final_rmse_scores.append(rmse_score)

        # --- 绘图 ---
        fig = plt.figure(figsize=(16, 9))
        # <--- MODIFIED: 在绘图标题中添加种子信息 ---
        fig.suptitle(f'Test Trajectory {i + 1} - True vs. Predicted (RMSE: {rmse_score:.4f})\n'
                     f"delay={params['delay_step']}, phi={params['PhiDimensions']}, seed={seed}")
        time_axis = np.arange(y_true.shape[1])

        for j in range(6):
            ax = plt.subplot(2, 3, j + 1)
            ax.plot(time_axis, y_true[j, :], 'b-', linewidth=1.5, label='True')
            ax.plot(time_axis, y_pred[j, :], 'r--', linewidth=1.5, label='Predicted')
            ax.set_title(f'Dimension {j + 1}')
            ax.set_xlabel('Time Step')
            ax.set_ylabel('Value')
            ax.grid(True)
            if j == 0:
                ax.legend(loc='upper right')

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        # <--- MODIFIED: 在绘图文件名中添加种子信息 ---
        plot_filename = model_save_path / f'test_trajectory_{i + 1}_seed_{seed}_comparison.png'
        plt.savefig(plot_filename)
        plt.close(fig)

    mean_final_rmse = np.mean(final_rmse_scores)
    print(
        f'\n--- [Seed: {seed}] 参数组合 (delay={params["delay_step"]}, phi={params["PhiDimensions"]}) 的最终平均RMSE: {mean_final_rmse:.6f} ---')
    return mean_final_rmse


def main_hyperparameter_search():
    """
    主函数，用于执行包含随机种子迭代的超参数搜索的整个流程。
    """
    # ==========================================================================
    # 1. 基础参数和路径设置
    # ==========================================================================
    print("## 1. 设置基础参数和路径... ##")
    # ... (这部分路径设置与您原版相同) ...
    current_dir = Path(__file__).resolve().parent
    base_data_path = current_dir.parent.parent / "data" / "SorotokiData" / "MotionData2" / "FilteredDataPos"
    train_path = base_data_path / "80minTrain"
    test_path = base_data_path / "50secTest"
    base_model_save_path = current_dir / "models" / "LKO_hyperparameter_search_robust" # 建议换个新名字
    base_model_save_path.mkdir(parents=True, exist_ok=True)
    print(f"所有模型和结果将保存在基础路径: '{base_model_save_path}'")

    control_var_name = 'input'
    state_var_name = 'state'
    is_norm = True

    base_params = {
        'is_norm': is_norm,
        'state_size': 6,
        'control_size': 6,
        'hidden_size_lstm': 128,
        'hidden_size_mlp': 128,
        'initialLearnRate': 5e-3,
        'minLearnRate': 1e-5,
        'num_epochs': 200,
        'L1': 1.0,
        'L2': 1.0,
        'L3': 0.0001,
        'batchSize': 8172,
        'patience': 1000,
        'lrReduceFactor': 0.2,
        'pred_step': 5
    }

    if torch.cuda.is_available():
        print('检测到可用GPU，启用加速')
        base_params['device'] = torch.device('cuda')
    else:
        print('未检测到GPU，使用CPU')
        base_params['device'] = torch.device('cpu')

    # <--- ADDED: 定义要迭代的随机种子列表 ---
    seed_list = [42, 123, 2024, 888, 1000, 5678, 99, 777, 4321, 654]
    print(f"将为每个超参数组合测试 {len(seed_list)} 个随机种子: {seed_list}")

    delay_step_range = range(9, 11)
    i_range = range(12, 31)

    all_results = []
    # <--- MODIFIED: 总运行次数是超参数组合的数量 ---
    total_hp_combinations = len(delay_step_range) * len(i_range)
    current_hp_run = 0

    # ==========================================================================
    # 2. 加载原始数据文件 (此部分与您原版相同)
    # ==========================================================================
    print("\n## 2. 加载所有原始 .mat 文件... ##")
    # ... (省略与您原版相同的代码) ...
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
    print("归一化参数计算完成。")
    raw_train_data = [{'control': loadmat(f)[control_var_name], 'state': loadmat(f)[state_var_name]} for f in train_files]
    raw_test_data = [{'control': loadmat(f)[control_var_name], 'state': loadmat(f)[state_var_name]} for f in test_files]

    # ==========================================================================
    # 3. 超参数搜索主循环
    # ==========================================================================
    start_time_total = time.time()

    for delay_step in delay_step_range:
        print(f"\n{'=' * 80}\n正在处理 delay_step = {delay_step}\n{'=' * 80}")
        # --- 3.1. 为当前的 delay_step 生成训练和测试数据 (此部分与您原版相同) ---
        print(f"## 为 delay_step = {delay_step} 生成数据... ##")
        # ... (省略与您原版相同的代码) ...
        control_train_list, state_train_list, label_train_list = [], [], []
        for raw_data in raw_train_data:
            state_data, control_data = raw_data['state'], raw_data['control']
            if is_norm:
                state_data, _ = normalize_data(state_data, params_state)
                control_data, _ = normalize_data(control_data, params_control)
            ctrl_td, state_td, label_td = generate_lstm_data(control_data, state_data, delay_step, base_params['pred_step'])
            control_train_list.append(ctrl_td)
            state_train_list.append(state_td)
            label_train_list.append(label_td)
        train_data = {'control': np.concatenate(control_train_list, axis=0), 'state': np.concatenate(state_train_list, axis=0), 'label': np.concatenate(label_train_list, axis=0)}
        print(f"训练数据已生成。总样本数: {train_data['state'].shape[0]}")
        test_data = []
        for raw_data in raw_test_data:
            control_test_raw, state_test_raw = raw_data['control'], raw_data['state']
            if is_norm:
                state_test_norm, _ = normalize_data(state_test_raw, params_state)
                control_test_norm, _ = normalize_data(control_test_raw, params_control)
            ctrl_td, state_td, label_td = generate_lstm_data(control_test_norm, state_test_norm, delay_step, base_params['pred_step'])
            test_data.append({'control': torch.from_numpy(ctrl_td).float().to(base_params['device']), 'state': torch.from_numpy(state_td).float().to(base_params['device']), 'label': torch.from_numpy(label_td).float().to(base_params['device'])})
        print(f"测试数据已生成。测试轨迹数量: {len(test_data)}")

        for i in i_range:
            current_hp_run += 1
            start_time_hp_set = time.time()

            params = base_params.copy()
            params['delay_step'] = delay_step
            params['PhiDimensions'] = delay_step * i
            params['output_size'] = params['PhiDimensions']

            run_folder_name = f"delay_{delay_step}_phi_{params['PhiDimensions']}"
            current_model_save_path = base_model_save_path / run_folder_name
            current_model_save_path.mkdir(parents=True, exist_ok=True)

            print(f"\n--- [{current_hp_run}/{total_hp_combinations}] 正在处理参数组合: {run_folder_name} ---")

            # <--- MODIFIED: 种子循环 ---
            rmse_scores_for_this_hp = []
            for seed in seed_list:
                # 1. 在每次训练前设置种子
                set_seed(seed)
                # 2. 将种子存入参数字典，以便传递给下游函数
                params['seed'] = seed
                # 3. 运行单次实验
                mean_rmse = run_experiment(params, train_data, test_data, is_norm, params_state, current_model_save_path)
                rmse_scores_for_this_hp.append(mean_rmse)

            # <--- MODIFIED: 计算并聚合结果 ---
            avg_rmse = np.mean(rmse_scores_for_this_hp)
            std_rmse = np.std(rmse_scores_for_this_hp)

            all_results.append({
                'delay_step': delay_step,
                'PhiDimensions': params['PhiDimensions'],
                'i_multiplier': i,
                'avg_rmse': avg_rmse,
                'std_rmse': std_rmse,
                'folder': run_folder_name
            })

            elapsed_time_hp_set = time.time() - start_time_hp_set
            elapsed_time_total = time.time() - start_time_total
            avg_time_per_hp_run = elapsed_time_total / current_hp_run
            remaining_hp_runs = total_hp_combinations - current_hp_run
            estimated_time_remaining = avg_time_per_hp_run * remaining_hp_runs

            print(f"\n--- 参数组合 {run_folder_name} 在 {len(seed_list)} 个种子上的聚合结果 ---")
            print(f"    平均 RMSE: {avg_rmse:.6f}")
            print(f"    RMSE 标准差: {std_rmse:.6f}")
            print(f"    本次参数组合总耗时: {elapsed_time_hp_set:.2f} 秒. 预计剩余时间: {estimated_time_remaining / 3600:.2f} 小时 ---")

    # ==========================================================================
    # 4. 总结和打印最终结果
    # ==========================================================================
    print(f"\n\n{'=' * 80}\n超参数搜索完成！\n{'=' * 80}")

    # <--- MODIFIED: 根据平均RMSE从小到大排序 ---
    sorted_results = sorted(all_results, key=lambda x: x['avg_rmse'])

    print("所有参数组合的测试损失（RMSE）排名 (根据平均RMSE排序，从小到大):")
    print("-" * 110)
    print(f"{'排名':<5} | {'delay_step':<12} | {'PhiDimensions':<15} | {'Avg RMSE':<18} | {'Std Dev':<18} | {'文件夹':<40}")
    print("-" * 110)
    for rank, result in enumerate(sorted_results, 1):
        print(
            f"{rank:<5} | {result['delay_step']:<12} | {result['PhiDimensions']:<15} | {result['avg_rmse']:.6f}{'':<12} | {result['std_rmse']:.6f}{'':<12} | {result['folder']}")
    print("-" * 110)

    if sorted_results:
        best_result = sorted_results[0]
        print("\n--- 最优参数组合 ---")
        print(f"文件夹: {best_result['folder']}")
        print(f"平均RMSE: {best_result['avg_rmse']:.6f}")
        print(f"RMSE标准差: {best_result['std_rmse']:.6f} (此值越小，模型越稳定)")


if __name__ == '__main__':
    main_hyperparameter_search()