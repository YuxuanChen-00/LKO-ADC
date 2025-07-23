import torch
import numpy as np
from scipy.io import loadmat
import matplotlib

# --- 设置后端，确保在无图形界面的服务器上也能运行 ---
# 'Agg' 是非交互式后端，只会将图像渲染到文件，不会尝试显示。
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

# --- 从您的项目中导入所需的函数和类 ---
from evaluate_lstm_lko import evaluate_lstm_lko, evaluate_lstm_lko2
from generate_lstm_data import generate_lstm_data
from src.normalize_data import normalize_data, denormalize_data
from src.DRKO.lko_lstm_network import LKO_lstm_Network
from train_lstm_lko import train_lstm_lko
from evaluate_lstm_lko import calculate_rmse

# --- 路径和参数设置 (与原版相同) ---
current_dir = Path(__file__).resolve().parent
base_data_path = current_dir.parent.parent / "data" / "SorotokiData" / "MotionData8" / "FilteredDataPos"
test_path = base_data_path / "50secTest"
train_path = base_data_path / "80minTrain"
model_path = (
    "/root/autodl-tmp/python_koopman_project/src/DRKO/models/LKO_lstm_SorotokiPositionData_network/trained_network.pth")

control_var_name = 'input'
state_var_name = 'state'
is_norm = True
params = {}
params['is_norm'] = is_norm
params['state_size'] = 6
params['delay_step'] = 8
params['control_size'] = 6
params['PhiDimensions'] = 6*18
params['hidden_size_lstm'] = 256
params['hidden_size_mlp'] = 64
params['output_size'] = params['PhiDimensions']
params['initialLearnRate'] = 0.005
params['minLearnRate'] = 1e-6
params['num_epochs'] = 200
params['L1'] = 1.0
params['L2'] = 1.0
params['L3'] = 0.0001
params['batchSize'] = 8172
params['patience'] = 1000
params['lrReduceFactor'] = 0.2
params['pred_step'] = 5
params['seed'] = 42

loss_pred_step = params['pred_step']

if torch.cuda.is_available():
    print('检测到可用GPU，启用加速')
    params['device'] = torch.device('cuda')
else:
    print('未检测到GPU，使用CPU')
    params['device'] = torch.device('cpu')
device = params['device']

# ==========================================================================
print("\n## 1. Calculating Normalization Parameters... ##")
train_files = sorted(list(train_path.glob('*.mat')))

state_for_norm = np.array([]).reshape(params['state_size'], 0)
control_for_norm = np.array([]).reshape(params['control_size'], 0)
for file_path in train_files:
    data = loadmat(file_path)
    state_for_norm = np.concatenate((state_for_norm, data[state_var_name]), axis=1)
    control_for_norm = np.concatenate((control_for_norm, data[control_var_name]), axis=1)

_, params_state = normalize_data(state_for_norm)
_, params_control = normalize_data(control_for_norm)
print("Normalization parameters calculated.")

# ==========================================================================
print("\n## 2. Loading and preprocessing test data... ##")
test_files = sorted(list(test_path.glob('*.mat')))
test_data = []

for file_path in test_files:
    data = loadmat(file_path)
    control_test = data[control_var_name]
    state_test = data[state_var_name]

    if is_norm:
        state_test, _ = normalize_data(state_test, params_state)
        control_test, _ = normalize_data(control_test, params_control)

    ctrl_td, state_td, label_td = generate_lstm_data(
        control_test, state_test, params['delay_step'], loss_pred_step
    )

    ctrl_td = torch.from_numpy(ctrl_td).float().to(device)
    state_td = torch.from_numpy(state_td).float().to(device)
    label_td = torch.from_numpy(label_td).float().to(device)

    test_data.append({
        'control': ctrl_td,
        'state': state_td,
        'label': label_td
    })
print(f"Test data processed. Number of test trajectories: {len(test_data)}")

# ==========================================================================
print("\n## 3. Loading LSTM-Koopman Model... ##")
state_size = params['state_size']
delay_step = params['delay_step']
hidden_size_lstm = params['hidden_size_lstm']
hidden_size_mlp = params['hidden_size_mlp']
output_size = params['output_size']
control_size = params['control_size']

net = LKO_lstm_Network(state_size, hidden_size_lstm, hidden_size_mlp, output_size, control_size, delay_step)
# 注意：确保你的模型路径是正确的
net.load_state_dict(torch.load(model_path, map_location=device))
net.eval()
net.to(device)

# ==========================================================================
print("\n## 4. Final evaluation and plotting... ##")

# ==================== 新增代码：创建用于保存图像的文件夹 ====================
plots_dir = current_dir / "prediction_plots"
plots_dir.mkdir(exist_ok=True)
print(f"预测结果图将保存在: {plots_dir.resolve()}")
# =======================================================================

final_rmse_scores = []

net.eval()
net.to(device)

for i, test_set in enumerate(test_data):
    control_test = test_set['control']
    state_test = test_set['state']
    label_test = test_set['label']
    initial_state_sequence = state_test[10 - params['delay_step'], :, :]

    current_file_name = test_files[i].name
    print("-" * 60)
    print(f"Processing Test File [{i + 1}/{len(test_data)}]: '{current_file_name}'")

    with torch.no_grad():
        rmse_score, y_true, y_pred = evaluate_lstm_lko2(net, control_test[10 - params['delay_step']:],
                                                       initial_state_sequence, label_test[10 - params['delay_step']:],
                                                       params_state, is_norm)

    final_rmse_scores.append(rmse_score)
    print(f"  -> RMSE: {rmse_score:.6f}")

    # --- Plotting ---
    fig = plt.figure(figsize=(16, 9))
    fig.suptitle(f'Test Trajectory {i + 1} - True vs. Predicted (RMSE: {rmse_score:.4f})')
    time_axis = np.arange(y_true.shape[1])


    for j in range(6):  # Plot first 6 dimensions
        ax = plt.subplot(2, 3, j + 1)
        ax.plot(time_axis, y_true[j, :], 'b-', linewidth=1.5, label='True')
        ax.plot(time_axis, y_pred[j, :], 'r--', linewidth=1.5, label='Predicted')
        ax.set_title(f'Dimension {j + 1}')
        ax.set_xlabel('Time Step')
        ax.set_ylabel('Value')
        ax.grid(True)
        if j == 0:
            ax.legend(loc='upper right')

    # 4. 调整布局并保存图像
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # 调整布局以适应主标题
    save_path = plots_dir / f"{Path(current_file_name).stem}_prediction.png"
    plt.savefig(save_path)
    plt.close(fig)  # 关闭画布，释放内存
    print(f"  -> 结果图已保存至: {save_path.name}")
    # =================================================================

# ==================== 新增代码：计算并打印平均损失 (格式调整) ====================
if final_rmse_scores:
    average_rmse = np.mean(final_rmse_scores)
    print("\n" + "=" * 60)
    print(f"✅ Final Result: Average RMSE across all {len(final_rmse_scores)} test files: {average_rmse:.6f}")
    print("=" * 60)
else:
    print("\nNo test files were evaluated.")
# =================================================================
