import torch
import numpy as np
from scipy.io import loadmat
import matplotlib

from src.DRKO.lko_lstm_network import LKO_lstm_Network

matplotlib.use('Agg')  # 或者 'TkAgg', 'Qt5Agg' 等，Agg 是非交互式后端，适合在无图形界面服务器上运行
import matplotlib.pyplot as plt
from pathlib import Path

# --- Import the functions and classes we previously converted ---
# These should be in your project directory

from evaluate_lstm_lko import evaluate_lstm_lko, evaluate_lstm_lko2
from generate_lstm_data import generate_lstm_data
from src.normalize_data import normalize_data, denormalize_data
from train_lstm_lko import train_lstm_lko
from evaluate_lstm_lko import calculate_rmse

current_dir = Path(__file__).resolve().parent
base_data_path = current_dir.parent.parent / "data" / "SorotokiData" / "MotionData8" / "FilteredDataPos"
test_path = base_data_path / "50secTest"
train_path = base_data_path / "80minTrain"
model_path = ("models/LKO_GridSearch_ray_Final_motion8/GridSearch_Delay_IMult_Seed"
              "/train_model_25104_00217_217_delay_step=6,i_multiplier=18,seed=666_2025-07-14_21-26-35/checkpoint_000000/final_model.pth")


control_var_name = 'input'
state_var_name = 'state'
is_norm = True
params = {}
params['is_norm'] = is_norm
params['state_size'] = 6
params['delay_step'] = 6
params['control_size'] = 6
params['PhiDimensions'] = 18*6
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


# 2. 计算归一化参数
# ==========================================================================
print("\n## 1. Calculating Normalization Parameters... ##")
train_files = sorted(list(train_path.glob('*.mat')))

# --- First pass: Calculate normalization parameters ---
state_for_norm = np.array([]).reshape(params['state_size'], 0)
control_for_norm = np.array([]).reshape(params['control_size'], 0)
for file_path in train_files:
    data = loadmat(file_path)
    state_for_norm = np.concatenate((state_for_norm, data[state_var_name]), axis=1)
    control_for_norm = np.concatenate((control_for_norm, data[control_var_name]), axis=1)

# Calculate and store normalization params
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

    # We need the full sequence for evaluation later
    test_data.append({
        'control': ctrl_td,
        'state': state_td,
        'label': label_td
    })
print(f"Test data processed. Number of test trajectories: {len(test_data)}")


print("\n## 3. Loading LSTM-Koopman Model... ##")
state_size = params['state_size']
delay_step = params['delay_step']
hidden_size_lstm = params['hidden_size_lstm']
hidden_size_mlp = params['hidden_size_mlp']
output_size = params['output_size']
control_size = params['control_size']

net = LKO_lstm_Network(state_size, hidden_size_lstm, hidden_size_mlp, output_size, control_size, delay_step)
net.load_state_dict(torch.load(model_path))
net.eval()
net.to(device)

print("\n## 4. Final evaluation and plotting... ##")
final_rmse_scores = []

net.eval()  # 设置为评估模式
net.to(device)

for i, test_set in enumerate(test_data):
    control_test = test_set['control']
    state_test = test_set['state']
    label_test = test_set['label']
    initial_state_sequence = state_test[10-params['delay_step'], :, :]

    # 调用评估函数
    with torch.no_grad():  # 评估时禁用梯度计算
        rmse_score, y_true, y_pred = evaluate_lstm_lko(net, control_test[10-params['delay_step']:],
                                                      initial_state_sequence, label_test[10-params['delay_step']:], params_state, is_norm)

    final_rmse_scores.append(rmse_score)

    # ==================== 新增代码：打印当前文件的损失 ====================
    # 使用 test_files 列表来获取当前测试文件的文件名
    current_file_name = test_files[i].name
    print(f"Test File [{i + 1}/{len(test_data)}] '{current_file_name}': RMSE = {rmse_score:.6f}")
    # =================================================================

# ==================== 新增代码：计算并打印平均损失 ====================
if final_rmse_scores:
    average_rmse = np.mean(final_rmse_scores)
    print("\n" + "="*50)
    print(f"✅ Final Result: Average RMSE across all {len(final_rmse_scores)} test files: {average_rmse:.6f}")
    print("="*50)
else:
    print("\nNo test files were evaluated.")
# =================================================================