import torch
import numpy as np
from scipy.io import loadmat
import matplotlib

matplotlib.use('Agg')  # 设置为非交互式后端
import matplotlib.pyplot as plt
from pathlib import Path

# --- 1. 导入新的、正确的模块和函数 ---
# 确保这些文件和函数在您的项目中存在且路径正确
from train_hybrid_ltv_model import train_hybrid_ltv_model, train_two_stage_hybrid_model, train_hybrid_ltv_model_e2e  # 导入我们新的训练函数
from evaluate_hybrid_model import evaluate_hybrid_ltv_model  # 导入我们新的评估函数
from dataloader import generate_koopman_data  # 数据生成函数保持不变
from src.normalize_data import normalize_data, denormalize_data
from src.calculate_rmse import calculate_rmse  # 假设RMSE计算函数不变


def main():
    """
    主函数，运行LTI基线模型的完整训练和评估流程。
    """
    # ==========================================================================
    # 1. 参数设置 (已更新以适配新模型)
    # ==========================================================================
    print("## 1. 设置参数... ##")
    # --- 路径设置 ---
    current_dir = Path(__file__).resolve().parent
    base_data_path = current_dir.parent.parent / "data" / "SorotokiData" / "MotionData8" / "FilteredDataPos"
    train_path = base_data_path / "80minTrain"
    test_path = base_data_path / "50secTest"
    model_save_path = current_dir / "models" / "Hybrid_LTV_Koopman"  # 建议为新模型使用新目录
    model_save_path.mkdir(parents=True, exist_ok=True)
    print(f"模型将保存在: '{model_save_path}'")

    # --- 数据和通用参数 ---
    control_var_name = 'input'
    state_var_name = 'state'

    # --- 神经网络参数 ---
    params = {}
    params['is_norm'] = True
    params['state_size'] = 6
    params['delay_step'] = 8
    params['control_size'] = 6
    params['pred_step'] = 1
    params['seed'] = 666

    # --- 新模型架构参数 ---
    params['PhiDimensions'] = 6 * 18
    params['encoder_gru_hidden'] = 256
    params['encoder_mlp_hidden'] = 128
    params['delta_rnn_hidden'] = 64  # 即使不训练，也需要定义以实例化模型
    params['delta_mlp_hidden'] = 64  # 同上

    # --- 训练超参数 ---
    params['initialLearnRate'] = 5e-4  # 为更复杂的模型使用一个稍小的学习率
    params['minLearnRate'] = 1e-6
    params['num_epochs'] = 100  # LTI模型可以训练更久以充分收敛
    params['batchSize'] = 1024
    params['patience'] = 1000

    # --- 损失函数权重 ---
    params['L1'] = 1  # 预测损失
    params['L2'] = 1  # 线性一致性损失
    params['L3'] = 100  # 重构损失
    params['L_delta'] = 1  # 在第一阶段，此项无效

    # --- 核心：设置训练模式为第一阶段 ---
    params['train_mode'] = 'stage1'
    params['pretrained_path'] = ''
    params['num_epochs_s1'] = 100
    params['lr_s1'] = 5e-5
    params['num_epochs_s2'] = 50
    params['lr_s2'] = 1e-5

    # --- 设备设置 ---
    if torch.cuda.is_available():
        print('检测到可用GPU，启用加速')
        params['device'] = torch.device('cuda')
    else:
        print('未检测到GPU，使用CPU')
        params['device'] = torch.device('cpu')

    # ==========================================================================
    # 2. 加载和预处理数据 (此部分逻辑与您提供的代码相同)
    # ==========================================================================
    print("\n## 2. 加载和预处理数据... ##")
    train_files = sorted(list(train_path.glob('*.mat')))
    test_files = sorted(list(test_path.glob('*.mat')))

    # 计算归一化参数
    state_for_norm = np.concatenate([loadmat(f)[state_var_name] for f in train_files], axis=1)
    _, params_state = normalize_data(state_for_norm)
    params['params_state'] = params_state
    print("归一化参数计算完毕。")

    # 定义一个内部函数来处理数据生成，避免代码重复
    def process_data_files(file_list, params):
        data_list = []
        for file_path in file_list:
            data = loadmat(file_path)
            state_data = data[state_var_name]
            control_data = data[control_var_name]

            if params['is_norm']:
                state_data, _ = normalize_data(state_data, params_state)

            # 假设控制数据不需要归一化
            ctrl_td, state_td, label_td = generate_koopman_data(control_data, state_data, params['delay_step'],
                                                             params['pred_step'])

            # 注意：对于最终评估，我们直接传递整个张量，而不是一个字典列表
            data_list.append({
                'control': torch.from_numpy(ctrl_td).float().to(params['device']),
                'state': torch.from_numpy(state_td).float().to(params['device']),
                'label': torch.from_numpy(label_td).float().to(params['device'])
            })
        return data_list

    # 将所有训练轨迹数据合并到一个字典中
    train_data_list = process_data_files(train_files, params)
    train_data = {
        'control': np.concatenate([d['control'].cpu().numpy() for d in train_data_list], axis=0),
        'state': np.concatenate([d['state'].cpu().numpy() for d in train_data_list], axis=0),
        'label': np.concatenate([d['label'].cpu().numpy() for d in train_data_list], axis=0)
    }
    print(f"训练数据处理完毕。总样本数: {train_data['state'].shape[0]}")

    test_data = process_data_files(test_files, params)
    print(f"测试数据处理完毕。测试轨迹数: {len(test_data)}")

    # ==========================================================================
    # 3. 训练LTI基线网络
    # ==========================================================================
    print("\n## 3. 开始网络训练 (阶段一: LTI基线模型)... ##")

    # 调用新的训练函数，它会根据 params['train_mode']='stage1' 自动冻结时变部分
    lti_net = train_two_stage_hybrid_model(params, train_data, test_data)

    # 保存第一阶段训练好的模型
    lti_model_path = model_save_path / 'lti_baseline_model.pth'
    torch.save(lti_net.state_dict(), lti_model_path)
    print(f"第一阶段训练完成。LTI基线模型已保存至 '{lti_model_path}'")

    # ==========================================================================
    # 4. 对LTI基线模型进行最终评估和绘图
    # ==========================================================================
    print("\n## 4. LTI基线模型最终评估与绘图... ##")
    final_rmse_scores = []
    lti_net.eval()
    lti_net.to(params['device'])

    for i, test_set in enumerate(test_data):
        control_test = test_set['control']
        state_test = test_set['state']
        label_test = test_set['label']

        # 定义评估的起始时间点
        start_index = 10 - params['delay_step']  # 定义一个固定的评估起始点

        # 1. 提取单个初始状态历史 (shape: [delay_step, state_size])
        initial_state_sequence = state_test[start_index]

        # 2. 提取单个初始控制历史 (shape: [delay_step, control_size])
        initial_control_sequence = control_test[start_index, 0]

        # 3. 提取从起始点开始的未来真实标签和控制
        #    这些将传递给评估函数用于计算RMSE和作为未来控制输入
        future_labels_for_eval = label_test[start_index:, 0]
        future_controls_for_eval = control_test[start_index:, 0]

        # 调用新的评估函数
        with torch.no_grad():
            rmse_score, y_true, y_pred = evaluate_hybrid_ltv_model(
                lti_net,
                initial_state_sequence,
                initial_control_sequence,
                future_labels_for_eval,
                future_controls_for_eval,
                params_state,
                params['is_norm']
            )

        final_rmse_scores.append(rmse_score)

        # --- 绘图 ---
        fig = plt.figure(figsize=(16, 9))
        fig.suptitle(f'LTI模型 - 测试轨迹 {i + 1} - 真实值 vs. 预测值 (RMSE: {rmse_score:.4f})')
        time_axis = np.arange(y_true.shape[1])
        for j in range(6):
            ax = plt.subplot(2, 3, j + 1)
            ax.plot(time_axis, y_true[j, :], 'b-', linewidth=1.5, label='True')
            ax.plot(time_axis, y_pred[j, :], 'r--', linewidth=1.5, label='Predicted')
            ax.set_title(f'维度 {j + 1}')
            ax.set_xlabel('Time Step')
            ax.grid(True)
            if j == 0: ax.legend()
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plot_filename = model_save_path / f'lti_model_trajectory_{i + 1}.png'
        plt.savefig(plot_filename)
        plt.close(fig)
        print(f"评估图已保存: {plot_filename}")

    mean_final_rmse = np.mean(final_rmse_scores)
    print(f'\n--- LTI基线模型在所有测试数据上的最终平均RMSE: {mean_final_rmse:.4f} ---')


if __name__ == '__main__':
    main()