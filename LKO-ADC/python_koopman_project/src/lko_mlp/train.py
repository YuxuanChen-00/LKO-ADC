import torch
import numpy as np
from scipy.io import loadmat
import matplotlib
matplotlib.use('Agg') # 或者 'TkAgg', 'Qt5Agg' 等，Agg 是非交互式后端，适合在无图形界面服务器上运行
import matplotlib.pyplot as plt
from pathlib import Path

# --- Import the functions and classes we previously converted ---
# These should be in your project directory

from evalutate_mlp_lko import evaluate_mlp_lko
from generate_mlp_data import generate_mlp_data
from src.normalize_data import normalize_data, denormalize_data
from  train_mlp_lko import train_mlp_lko
from evalutate_mlp_lko import calculate_rmse

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
    base_data_path = current_dir.parent.parent / "data" / "SorotokiData" / "MotionData7" / "FilteredDataPos"

    train_path = base_data_path / "80minTrain"


    test_path = base_data_path / "50secTest"
    model_save_path = current_dir / "models" / "LKO_mlp_SorotokiPositionData_network"

    # Create model save directory if it doesn't exist
    model_save_path.mkdir(parents=True, exist_ok=True)
    print(f"Model will be saved in: '{model_save_path}'")

    # --- Data Generation Parameters ---
    control_var_name = 'input'
    state_var_name = 'state'
    loss_pred_step = 1
    is_norm = False

    # --- Neural Network Parameters ---
    params = {}
    params['state_size'] = 6
    params['delay_step'] = 3
    params['control_size'] = 6
    params['PhiDimensions'] = 27
    params['hidden_size'] = int((params['PhiDimensions'] + params['state_size']) / 2)
    params['output_size'] = params['PhiDimensions'] - params['state_size']
    params['initialLearnRate'] = 0.01
    params['minLearnRate'] = 1e-6
    params['num_epochs'] = 500
    params['L1'] = 100.0
    params['L2'] = 1.0
    params['L3'] = 0.0001
    params['batchSize'] = 1024
    params['patience'] = 200
    params['lrReduceFactor'] = 0.2

    # Add pred_step to params for use in util functions
    params['pred_step'] = loss_pred_step

    # ==========================================================================
    # 2. Load and Preprocess Training Data
    # ==========================================================================
    print("\n## 2. Loading and preprocessing training data... ##")
    train_files = sorted(list(train_path.glob('*.mat')))

    # --- First pass: Calculate normalization parameters ---
    state_for_norm = np.array([]).reshape(params['state_size'], 0)
    for file_path in train_files:
        data = loadmat(file_path)
        state_for_norm = np.concatenate((state_for_norm, data[state_var_name]), axis=1)

    # Calculate and store normalization params
    _, params_state = normalize_data(state_for_norm)
    print("Normalization parameters calculated.")

    # --- Second pass: Process and aggregate training data ---
    control_train_list, state_train_list, label_train_list = [], [], []
    for file_path in train_files:
        data = loadmat(file_path)

        state_data = data[state_var_name]
        if is_norm:
            state_data, _ = normalize_data(state_data, params_state)

        ctrl_td, state_td, label_td = generate_mlp_data(
            data[control_var_name], state_data, params['delay_step'], loss_pred_step
        )
        control_train_list.append(ctrl_td)
        state_train_list.append(state_td)
        label_train_list.append(label_td)

    # Concatenate data from all files
    # Note: MATLAB's cat(2,...) on 3D arrays is equivalent to numpy's concatenate on axis=2
    # But our generate_mlp_data has a different dimension order, so we adjust.
    control_train = np.concatenate(control_train_list, axis=2)
    state_train = np.concatenate(state_train_list, axis=1)
    label_train = np.concatenate(label_train_list, axis=2)

    train_data = {
        'control': control_train,
        'state': state_train,
        'label': label_train
    }
    print(f"Training data processed. Total samples: {state_train.shape[1]}")

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

        # We need the full sequence for evaluation later
        test_data.append({
            'control': control_test,
            'state': state_test
        })
    print(f"Test data processed. Number of test trajectories: {len(test_data)}")

    # ==========================================================================
    # 4. Train the Network
    # ==========================================================================
    print("\n## 4. Starting network training... ##")
    # The train_mlp_lko function handles the full training loop
    net = train_mlp_lko(params, train_data, test_data)

    # Save the final trained model
    final_model_path = model_save_path / 'trained_network.pth'
    torch.save(net.state_dict(), final_model_path)
    print(f"Training complete. Final model saved to '{final_model_path}'")

    # ==========================================================================
    # 5. Final Evaluation and Plotting
    # ==========================================================================
    print("\n## 5. Final evaluation and plotting... ##")
    final_rmse_scores = []

    for i, test_set in enumerate(test_data):
        control_test_raw = test_set['control']
        state_test_raw = test_set['state']
        # label_test_raw = test_set['label'] # label在评估时从state_test_raw中导出

        # 为评估准备数据
        pred_step_eval = control_test_raw.shape[1] - params['delay_step']  # 预测步数由测试控制信号的长度决定
        control_test_raw = control_test_raw[:, 0:pred_step_eval]
        initial_state_eval = torch.from_numpy(state_test_raw[:, :params['delay_step']]).unsqueeze(0)
        initial_state_eval = initial_state_eval.permute(0, 2, 1).float()  # (1, time_step, d)

        control_eval = torch.from_numpy(control_test_raw).permute(1, 0).float()  # (pred_step, c)

        true_labels_eval = np.zeros([params['state_size'], pred_step_eval])
        for k in range(pred_step_eval):
            true_labels_eval[:, k] = state_test_raw[:, k + params['delay_step']]

        true_labels_eval = torch.from_numpy(true_labels_eval)

        # 调用评估函数
        *_, y_true, y_pred = evaluate_mlp_lko(net, control_eval, initial_state_eval, true_labels_eval, params['delay_step'])

        # Denormalize for final comparison
        if is_norm:
            y_pred = denormalize_data(y_pred, params_state)
            y_true = denormalize_data(y_true, params_state)

        rmse_score = np.sqrt(np.mean((y_true - y_pred)**2))
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