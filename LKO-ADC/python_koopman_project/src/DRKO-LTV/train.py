import torch
import numpy as np
from scipy.io import loadmat
import matplotlib

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


# --- Normalization Helper Functions ---


def main():
    """
    The main function to run the entire training and evaluation pipeline.
    """
    # ==========================================================================
    # 1. Parameter Setup
    # ==========================================================================
    print("## 1. Setting up parameters... ##")
    # --- Paths Setup ---
    # Use pathlib for robust path handling
    current_dir = Path(__file__).resolve().parent
    base_data_path = current_dir.parent.parent / "data" / "SorotokiData" / "MotionData8" / "FilteredDataPos"

    train_path = base_data_path / "80minTrain"

    test_path = base_data_path / "50secTest"
    model_save_path = current_dir / "models" / "LKO_lstm_SorotokiPositionData_network"

    # Create model save directory if it doesn't exist
    model_save_path.mkdir(parents=True, exist_ok=True)
    print(f"Model will be saved in: '{model_save_path}'")

    # --- Data Generation Parameters ---
    control_var_name = 'input'
    state_var_name = 'state'
    is_norm = True

    # --- Neural Network Parameters ---
    params = {}
    params['is_norm'] = is_norm
    params['state_size'] = 6
    params['delay_step'] = 8
    params['control_size'] = 6
    params['PhiDimensions'] = 8*12
    params['hidden_size_lstm'] = 256
    params['hidden_size_mlp'] = 64
    params['output_size'] = params['PhiDimensions']
    params['initialLearnRate'] = 0.005
    params['minLearnRate'] = 1e-6
    params['num_epochs'] = 100
    params['L1'] = 1.0
    params['L2'] = 1.0
    params['L3'] = 100.0
    params['batchSize'] = 1024
    params['patience'] = 1000
    params['lrReduceFactor'] = 0.2
    params['pred_step'] = 10
    params['seed'] = 666

    loss_pred_step = params['pred_step']

    if torch.cuda.is_available():
        print('检测到可用GPU，启用加速')
        params['device'] = torch.device('cuda')
    else:
        print('未检测到GPU，使用CPU')
        params['device'] = torch.device('cpu')
    device = params['device']

    # Add pred_step to params for use in util functions

    # ==========================================================================
    # 2. Load and Preprocess Training Data
    # ==========================================================================
    print("\n## 2. Loading and preprocessing training data... ##")
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
    params['params_state'] = params_state
    params['params_control'] = params_control
    print("Normalization parameters calculated.")

    # --- Second pass: Process and aggregate training data ---
    control_train_list, state_train_list, label_train_list = [], [], []
    for file_path in train_files:
        data = loadmat(file_path)

        state_data = data[state_var_name]
        control_data = data[control_var_name]

        if is_norm:
            state_data, _ = normalize_data(state_data, params_state)
            control_data, _ = normalize_data(control_data, params_control)

        ctrl_td, state_td, label_td = generate_lstm_data(control_data, state_data, params['delay_step'], loss_pred_step)
        control_train_list.append(ctrl_td)
        state_train_list.append(state_td)
        label_train_list.append(label_td)

    # Concatenate data from all files
    # Note: MATLAB's cat(2,...) on 3D arrays is equivalent to numpy's concatenate on axis=2
    # But our generate_lstm_data has a different dimension order, so we adjust.
    control_train = np.concatenate(control_train_list, axis=0)
    state_train = np.concatenate(state_train_list, axis=0)
    label_train = np.concatenate(label_train_list, axis=0)

    # print(control_train.shape, state_train.shape, label_train.shape)

    train_data = {
        'control': control_train,
        'state': state_train,
        'label': label_train
    }
    print(f"Training data processed. Total samples: {state_train.shape[0]}")

    # ==========================================================================
    # 3. Load and Preprocess Test Data
    # ==========================================================================
    print("\n## 3. Loading and preprocessing test data... ##")
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

    # ==========================================================================
    # 4. Train the Network
    # ==========================================================================
    print("\n## 4. Starting network training... ##")
    # The train_lstm_lko function handles the full training loop
    net = train_lstm_lko(params, train_data, test_data)

    # Save the final trained model
    final_model_path = model_save_path / 'trained_network.pth'
    torch.save(net.state_dict(), final_model_path)
    print(f"Training complete. Final model saved to '{final_model_path}'")

    # ==========================================================================
    # 5. Final Evaluation and Plotting
    # ==========================================================================
    print("\n## 5. Final evaluation and plotting... ##")
    final_rmse_scores = []

    net.eval()  # 设置为评估模式
    test_loss_list = []

    # test_data 是一个字典列表
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

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plot_filename = model_save_path / f'test_trajectory_{i + 1}_comparison.png'
        # plt.show()
        plt.savefig(plot_filename)
        plt.close(fig)  # Close the figure to avoid displaying it in interactive environments
        print(f"Plot saved for trajectory {i + 1}: {plot_filename}")

    mean_final_rmse = np.mean(final_rmse_scores)
    print(f'\n--- Final Average RMSE on all test data: {mean_final_rmse:.4f} ---')


if __name__ == '__main__':
    main()
