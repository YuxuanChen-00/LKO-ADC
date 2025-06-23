import torch
import numpy as np
from scipy.io import loadmat
import matplotlib

# 使用非交互式后端，避免在服务器上弹出图形窗口
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
import time  # 引入 time 模块以估算剩余时间

# --- 假设这些自定义函数和类位于您的项目目录中 ---
# 请确保这些文件的路径是正确的
from evaluate_lstm_lko import evaluate_lstm_lko, evaluate_lstm_lko2
from generate_lstm_data import generate_lstm_data
from src.normalize_data import normalize_data, denormalize_data
from train_lstm_lko import train_lstm_lko
from evaluate_lstm_lko import calculate_rmse


def run_experiment(params, train_data, test_data, is_norm, params_state, model_save_path):
    """
    运行一次完整的训练和评估流程。

    Args:
        params (dict): 包含所有网络和训练参数的字典。
        train_data (dict): 包含训练数据的字典 ('control', 'state', 'label')。
        test_data (list): 包含多个测试轨迹字典的列表。
        is_norm (bool): 是否对数据进行了归一化。
        params_state (tuple): 状态数据的归一化参数 (mean, std)。
        model_save_path (Path): 用于保存此实验的模型和图片的文件夹路径。

    Returns:
        float: 在所有测试数据上的平均 RMSE。
    """
    print(f"\n--- 开始训练: delay_step={params['delay_step']}, PhiDimensions={params['PhiDimensions']} ---")

    # ==========================================================================
    # 4. 训练网络
    # ==========================================================================
    net = train_lstm_lko(params, train_data, test_data)

    # 为模型文件名添加参数信息
    model_filename = f"model_delay_{params['delay_step']}_phi_{params['PhiDimensions']}.pth"
    final_model_path = model_save_path / model_filename
    torch.save(net.state_dict(), final_model_path)
    print(f"训练完成。模型已保存至: '{final_model_path}'")

    # ==========================================================================
    # 5. 最终评估和绘图
    # ==========================================================================
    print("\n## 正在进行最终评估和绘图... ##")
    final_rmse_scores = []
    net.eval()  # 设置为评估模式
    device = params['device']
    net.to(device)

    for i, test_set in enumerate(test_data):
        delay_step = params['delay_step']
        control_test = test_set['control']
        state_test = test_set['state']
        label_test = test_set['label']
        initial_state_sequence = state_test[10 - delay_step, :, :]
        with torch.no_grad():
            rmse_score, y_true, y_pred = evaluate_lstm_lko(net, control_test[10 - delay_step:], initial_state_sequence,
                                                label_test[10 - delay_step:], params_state, is_norm)


        final_rmse_scores.append(rmse_score)

        # --- 绘图 ---
        fig = plt.figure(figsize=(16, 9))
        fig.suptitle(f'Test Trajectory {i + 1} - True vs. Predicted (RMSE: {rmse_score:.4f})\n'
                     f"delay_step={params['delay_step']}, PhiDimensions={params['PhiDimensions']}")
        time_axis = np.arange(y_true.shape[1])

        for j in range(6):  # 绘制前6个维度
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
        plot_filename = model_save_path / f'test_trajectory_{i + 1}_comparison.png'
        plt.savefig(plot_filename)
        plt.close(fig)
        print(f"已为轨迹 {i + 1} 保存绘图: {plot_filename}")

    mean_final_rmse = np.mean(final_rmse_scores)
    print(
        f'\n--- 参数组合 (delay={params["delay_step"]}, phi={params["PhiDimensions"]}) 的最终平均RMSE: {mean_final_rmse:.6f} ---')
    return mean_final_rmse


def main_hyperparameter_search():
    """
    主函数，用于执行超参数搜索的整个流程。
    """
    # ==========================================================================
    # 1. 基础参数和路径设置
    # ==========================================================================
    print("## 1. 设置基础参数和路径... ##")
    current_dir = Path(__file__).resolve().parent
    base_data_path = current_dir.parent.parent / "data" / "SorotokiData" / "MotionData2" / "FilteredDataPos"
    train_path = base_data_path / "80minTrain"
    test_path = base_data_path / "50secTest"
    # 基础模型保存路径，每个参数组合将在此下创建子文件夹
    base_model_save_path = current_dir / "models" / "LKO_hyperparameter_search_denorm_eval"
    base_model_save_path.mkdir(parents=True, exist_ok=True)
    print(f"所有模型和结果将保存在基础路径: '{base_model_save_path}'")

    control_var_name = 'input'
    state_var_name = 'state'
    is_norm = True

    # --- 基础神经网络参数 (可变参数将在循环中设置) ---
    base_params = {}
    base_params['is_norm'] = is_norm
    base_params['state_size'] = 6
    base_params['control_size'] = 6
    base_params['hidden_size_lstm'] = 128
    base_params['hidden_size_mlp'] = 128
    base_params['initialLearnRate'] = 5e-3
    base_params['minLearnRate'] = 1e-5
    base_params['num_epochs'] = 200  # 建议在搜索时减少epoch以加快速度, e.g., 50-100
    base_params['L1'] = 1.0
    base_params['L2'] = 1.0
    base_params['L3'] = 0.0001
    base_params['batchSize'] = 8172
    base_params['patience'] = 1000  # 搜索时可以适当减少
    base_params['lrReduceFactor'] = 0.2
    base_params['pred_step'] = 5

    if torch.cuda.is_available():
        print('检测到可用GPU，启用加速')
        base_params['device'] = torch.device('cuda')
    else:
        print('未检测到GPU，使用CPU')
        base_params['device'] = torch.device('cpu')

    # 定义超参数的搜索范围
    delay_step_range = range(9, 11)
    i_range = range(12, 31)

    # 用于存储所有实验结果
    all_results = []

    # 估算总运行次数
    total_runs = len(delay_step_range) * len(i_range)
    current_run = 0

    # ==========================================================================
    # 2. 加载原始数据文件
    # ==========================================================================
    print("\n## 2. 加载所有原始 .mat 文件... ##")
    train_files = sorted(list(train_path.glob('*.mat')))
    test_files = sorted(list(test_path.glob('*.mat')))

    # --- 第一遍: 计算归一化参数 ---
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
    print("归一化参数计算完成。")

    # --- 缓存原始数据到内存 ---
    raw_train_data = [{'control': loadmat(f)[control_var_name], 'state': loadmat(f)[state_var_name]} for f in
                      train_files]
    raw_test_data = [{'control': loadmat(f)[control_var_name], 'state': loadmat(f)[state_var_name]} for f in test_files]

    # ==========================================================================
    # 3. 超参数搜索主循环
    # ==========================================================================
    start_time_total = time.time()

    # 外层循环: 遍历 delay_step
    for delay_step in delay_step_range:
        print(f"\n{'=' * 60}\n正在处理 delay_step = {delay_step}\n{'=' * 60}")

        # --- 3.1. 为当前的 delay_step 生成训练和测试数据 ---
        print(f"## 为 delay_step = {delay_step} 生成数据... ##")

        # 处理训练数据
        control_train_list, state_train_list, label_train_list = [], [], []
        for raw_data in raw_train_data:
            state_data = raw_data['state']
            control_data = raw_data['control']

            if is_norm:
                state_data, _ = normalize_data(state_data, params_state)
                control_data, _ = normalize_data(control_data, params_control)

            ctrl_td, state_td, label_td = generate_lstm_data(control_data, state_data, delay_step,
                                                             base_params['pred_step'])
            control_train_list.append(ctrl_td)
            state_train_list.append(state_td)
            label_train_list.append(label_td)

        control_train = np.concatenate(control_train_list, axis=0)
        state_train = np.concatenate(state_train_list, axis=0)
        label_train = np.concatenate(label_train_list, axis=0)

        train_data = {'control': control_train, 'state': state_train, 'label': label_train}
        print(f"训练数据已生成。总样本数: {state_train.shape[0]}")

        # 处理测试数据
        test_data = []
        for raw_data in raw_test_data:
            control_test_raw = raw_data['control']
            state_test_raw = raw_data['state']

            if is_norm:
                state_test_norm, _ = normalize_data(state_test_raw, params_state)
                control_test_norm, _ = normalize_data(control_test_raw, params_control)

            ctrl_td, state_td, label_td = generate_lstm_data(control_test_norm, state_test_norm, delay_step,
                                                             base_params['pred_step'])

            test_data.append({
                'control': torch.from_numpy(ctrl_td).float().to(base_params['device']),
                'state': torch.from_numpy(state_td).float().to(base_params['device']),
                'label': torch.from_numpy(label_td).float().to(base_params['device'])
            })
        print(f"测试数据已生成。测试轨迹数量: {len(test_data)}")

        # 内层循环: 遍历 i 来计算 PhiDimensions
        for i in i_range:
            current_run += 1
            start_time_run = time.time()

            # --- 3.2. 设置当前运行的参数 ---
            params = base_params.copy()
            params['delay_step'] = delay_step
            params['PhiDimensions'] = delay_step * i
            params['output_size'] = params['PhiDimensions']

            # --- 3.3. 创建本次运行的专属文件夹 ---
            run_folder_name = f"delay_{delay_step}_phi_{params['PhiDimensions']}"
            current_model_save_path = base_model_save_path / run_folder_name
            current_model_save_path.mkdir(parents=True, exist_ok=True)

            print(f"\n--- [{current_run}/{total_runs}] 正在运行: {run_folder_name} ---")

            # --- 3.4. 运行训练和评估 ---
            mean_rmse = run_experiment(params, train_data, test_data, is_norm, params_state, current_model_save_path)

            # --- 3.5. 记录结果 ---
            all_results.append({
                'delay_step': delay_step,
                'PhiDimensions': params['PhiDimensions'],
                'i_multiplier': i,
                'loss_rmse': mean_rmse,
                'folder': run_folder_name
            })

            # --- 3.6. 估算剩余时间 ---
            elapsed_time_run = time.time() - start_time_run
            elapsed_time_total = time.time() - start_time_total
            avg_time_per_run = elapsed_time_total / current_run
            remaining_runs = total_runs - current_run
            estimated_time_remaining = avg_time_per_run * remaining_runs
            print(
                f"--- 本次运行耗时: {elapsed_time_run:.2f} 秒. 预计剩余时间: {estimated_time_remaining / 3600:.2f} 小时 ---")

    # ==========================================================================
    # 4. 总结和打印最终结果
    # ==========================================================================
    print(f"\n\n{'=' * 60}\n超参数搜索完成！\n{'=' * 60}")

    # 根据损失（RMSE）从大到小排序
    sorted_results = sorted(all_results, key=lambda x: x['loss_rmse'], reverse=True)

    print("所有参数组合的测试损失（RMSE）排名 (从高到低):")
    print("-" * 80)
    print(f"{'排名':<5} | {'delay_step':<12} | {'PhiDimensions':<15} | {'Loss (RMSE)':<18} | {'文件夹':<40}")
    print("-" * 80)
    for rank, result in enumerate(sorted_results, 1):
        print(
            f"{rank:<5} | {result['delay_step']:<12} | {result['PhiDimensions']:<15} | {result['loss_rmse']:.6f}{'':<12} | {result['folder']}")
    print("-" * 80)


if __name__ == '__main__':
    # 调用新的主函数
    main_hyperparameter_search()
