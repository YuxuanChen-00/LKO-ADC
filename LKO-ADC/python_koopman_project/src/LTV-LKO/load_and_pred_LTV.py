import torch
import numpy as np
from scipy.io import loadmat
import matplotlib
matplotlib.use('Agg')  # 设置为非交互式后端，以便在服务器上保存图片
import matplotlib.pyplot as plt
from pathlib import Path
import os

# --- 1. 导入您项目中的核心模块 ---
# 确保这些文件和函数在您的项目中存在且路径正确
from model_hybrid_ltv import HybridLTVKoopmanNetwork
from evaluate_hybrid_model import evaluate_hybrid_ltv_model
from dataloader import generate_koopman_data
from src.normalize_data import normalize_data

def predict_and_visualize():
    """
    主函数：加载Hybrid LTV Koopman模型，进行预测、评估并可视化结果。
    """
    # ==========================================================================
    # 1. 参数和路径设置 (与训练时保持一致)
    # ==========================================================================
    print("## 1. 设置参数与路径... ##")
    # --- 路径设置 ---
    current_dir = Path(__file__).resolve().parent
    base_data_path = current_dir.parent.parent / "data" / "SorotokiData" / "MotionData8" / "FilteredDataPos"
    train_path = base_data_path / "80minTrain"
    test_path = base_data_path / "50secTest"

    # --- ！！！核心：请将此处替换为您要加载的模型文件的确切路径！！！ ---
    model_load_path = ("/root/autodl-tmp/python_koopman_project/src/LTV-LKO/models/BOHB_Search_TwoStage_Final"
                       "/BOHB_Search_FullArch_Final/trial_f5c21c7e_8_L1=0.8845,L2=0.2802,L3=37.2820,L_delta=0.6211,"
                       "delay_step=14,delta_mlp_hidden=128,delta_rnn_hidden=64,"
                       "encoder_gru__2025-07-17_19-03-35/checkpoint_000000/final_model.pth")
    # 例如: model_load_path = current_dir / "models" / "Hybrid_LTV_Koopman" / "final_ltv_model.pth"

    # --- 创建保存结果的文件夹 ---
    results_save_path = current_dir / "results" / "Hybrid_LTV_Prediction"
    results_save_path.mkdir(parents=True, exist_ok=True)
    print(f"预测结果与图片将保存在: '{results_save_path}'")

    # --- 模型和数据参数 (必须与训练时完全一致) ---
    params = {
        'is_norm': True, 'state_size': 6, 'control_size': 6,
        'batchSize': 1024, 'seed': 666, 'minLearnRate': 1e-7,
        'num_epochs_s1': 150,
        'num_epochs_s2': 50,
        "L1": 0.8845087166352,
        "L2": 0.2801810386549,
        "L3": 37.2820093749576,
        "L_delta": 0.6210900101005,
        "delay_step": 14,
        "delta_mlp_hidden": 128,
        "delta_rnn_hidden": 64,
        "encoder_gru_hidden": 256,
        "encoder_mlp_hidden": 64,
        "i_multiplier": 28,
        "lr_s1": 3.36432616e-05,
        "lr_s2": 1.0605152e-06,
        "pred_step": 2
    }
    params['PhiDimensions'] = params['i_multiplier'] * params['delay_step']

    # --- 设备设置 ---
    if torch.cuda.is_available():
        print('检测到可用GPU，启用加速')
        params['device'] = torch.device('cuda')
    else:
        print('未检测到GPU，使用CPU')
        params['device'] = torch.device('cpu')
    device = params['device']

    # ==========================================================================
    # 2. 加载和预处理数据 (逻辑与 main.py 相同)
    # ==========================================================================
    print("\n## 2. 加载和预处理数据... ##")
    train_files = sorted(list(train_path.glob('*.mat')))
    test_files = sorted(list(test_path.glob('*.mat')))
    state_var_name = 'state'
    control_var_name = 'input'

    # --- 使用训练数据计算归一化参数 ---
    state_for_norm = np.concatenate([loadmat(f)[state_var_name] for f in train_files], axis=1)
    _, params_state = normalize_data(state_for_norm)
    params['params_state'] = params_state
    print("归一化参数计算完毕。")

    # --- 处理测试数据 ---
    test_data = []
    for file_path in test_files:
        data = loadmat(file_path)
        state_raw = data[state_var_name]
        control_raw = data[control_var_name]

        if params['is_norm']:
            state_normalized, _ = normalize_data(state_raw, params_state)

        # 生成适用于模型的序列数据
        ctrl_td, state_td, label_td = generate_koopman_data(control_raw, state_normalized, params['delay_step'], params['pred_step'])

        test_data.append({
            'control': torch.from_numpy(ctrl_td).float().to(device),
            'state': torch.from_numpy(state_td).float().to(device),
            'label': torch.from_numpy(label_td).float().to(device),
            'filename': file_path.name
        })
    print(f"测试数据处理完毕。共 {len(test_data)} 个测试轨迹。")

    # ==========================================================================
    # 3. 加载预训练模型
    # ==========================================================================
    print(f"\n## 3. 从 '{model_load_path}' 加载预训练模型... ##")
    if not os.path.exists(model_load_path):
        print(f"错误：找不到模型文件 '{model_load_path}'。请检查路径是否正确。")
        return

    # --- 初始化模型架构 ---
    net = HybridLTVKoopmanNetwork(
        state_size=params['state_size'],
        control_size=params['control_size'],
        time_step=params['delay_step'],
        g_dim=params['PhiDimensions'],
        encoder_gru_hidden=params['encoder_gru_hidden'],
        encoder_mlp_hidden=params['encoder_mlp_hidden'],
        delta_rnn_hidden=params['delta_rnn_hidden'],
        delta_mlp_hidden=params['delta_mlp_hidden']
    )
    # --- 加载模型权重 ---
    net.load_state_dict(torch.load(model_load_path, map_location=device))
    net.to(device)
    net.eval()  # !!! 必须设置为评估模式 !!!
    print("模型加载成功并已设置为评估模式。")

    # ==========================================================================
    # 4. 在测试集上进行最终评估和绘图
    # ==========================================================================
    print("\n## 4. 开始最终评估与绘图... ##")
    final_rmse_scores = []

    with torch.no_grad(): # 在评估过程中不计算梯度
        for i, test_set in enumerate(test_data):
            control_test = test_set['control']
            state_test = test_set['state']
            label_test = test_set['label']

            # 定义一个固定的评估起始点 (与main.py中评估逻辑一致)
            start_index = 10 - params['delay_step']
            if start_index < 0:
                print(f"警告: 测试文件 '{test_set['filename']}' 过短，跳过评估。")
                continue

            # 提取评估所需的序列
            initial_state_sequence = state_test[start_index]
            initial_control_sequence = control_test[start_index, 0]
            future_labels_for_eval = label_test[start_index:, 0]
            future_controls_for_eval = control_test[start_index:, 0]

            # 调用评估函数
            rmse_score, y_true, y_pred = evaluate_hybrid_ltv_model(
                net,
                initial_state_sequence,
                initial_control_sequence,
                future_labels_for_eval,
                future_controls_for_eval,
                params['params_state'],
                params['is_norm']
            )

            final_rmse_scores.append(rmse_score)
            print(f"测试文件 [{i+1}/{len(test_data)}] '{test_set['filename']}': RMSE = {rmse_score:.6f}")

            # --- 绘图 ---
            fig = plt.figure(figsize=(18, 10))
            fig.suptitle(f'Hybrid LTV Model Prediction vs. Ground Truth\nFile: {test_set["filename"]} | RMSE: {rmse_score:.4f}', fontsize=16)
            time_axis = np.arange(y_true.shape[1])
            for j in range(params['state_size']):
                ax = plt.subplot(3, 2, j + 1)
                ax.plot(time_axis, y_true[j, :], 'b-', linewidth=2, label='Ground Truth')
                ax.plot(time_axis, y_pred[j, :], 'r--', linewidth=2, label='Prediction')
                ax.set_title(f'State Dimension {j + 1}')
                ax.set_xlabel('Time Step')
                ax.set_ylabel('Value')
                ax.grid(True, linestyle='--', alpha=0.6)
                if j == 0:
                    ax.legend()
            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            plot_filename = results_save_path / f'prediction_{Path(test_set["filename"]).stem}.png'
            plt.savefig(plot_filename)
            plt.close(fig) # 关闭图形，释放内存

    # ==========================================================================
    # 5. 计算并打印平均性能
    # ==========================================================================
    if final_rmse_scores:
        mean_final_rmse = np.mean(final_rmse_scores)
        print("\n" + "="*60)
        print(f"✅ 评估完成: 在 {len(final_rmse_scores)} 个测试文件上的平均RMSE为: {mean_final_rmse:.6f}")
        print("="*60)
    else:
        print("\n没有有效的测试文件被评估。")


if __name__ == '__main__':
    predict_and_visualize()