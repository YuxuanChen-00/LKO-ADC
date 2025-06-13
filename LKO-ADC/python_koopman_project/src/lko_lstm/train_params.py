import torch
import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt
from pathlib import Path
import os
import pandas as pd
import shutil
from evalutate_lstm_lko import evaluate_lstm_lko
from generate_lstm_data import generate_lstm_data
from src.normalize_data import normalize_data, denormalize_data
from train_lstm_lko import train_lstm_lko
from evalutate_lstm_lko import calculate_rmse



def main():
    """
    The main function to run the entire training and evaluation pipeline with parameter iteration.
    """
    # ==========================================================================
    # 1. Parameter Setup
    # ==========================================================================
    print("## 1. Setting up parameters... ##")
    # --- Paths Setup ---
    current_dir = Path(__file__).resolve().parent
    base_data_path = current_dir.parent.parent / "data" / "SorotokiData" / "MotionData7" / "FilteredDataPos"
    train_path = base_data_path / "80minTrain"
    test_path = base_data_path / "50secTest"
    base_model_save_path = current_dir / "models" / "LKO_LSTM_SorotokiPositionData_network"
    results_save_path = current_dir / "results"

    # Create directories if they don't exist
    base_model_save_path.mkdir(parents=True, exist_ok=True)
    results_save_path.mkdir(parents=True, exist_ok=True)
    print(f"Models will be saved in: '{base_model_save_path}'")
    print(f"Results will be saved in: '{results_save_path}'")

    # --- Data Generation Parameters ---
    control_var_name = 'input'
    state_var_name = 'state'
    loss_pred_step = 1
    is_norm = False

    # --- Parameter ranges to iterate over ---
    delay_steps = range(1, 11)  # 1 to 10
    phi_dimensions = range(7, 31)  # 7 to 30

    # --- Results storage ---
    results = []

    # ==========================================================================
    # 2. Load and Preprocess Data (only once, before parameter iteration)
    # ==========================================================================
    print("\n## 2. Loading and preprocessing data... ##")
    # Load training files
    train_files = sorted(list(train_path.glob('*.mat')))

    # --- First pass: Calculate normalization parameters ---
    state_for_norm = np.array([]).reshape(6, 0)  # state_size is fixed at 6
    for file_path in train_files:
        data = loadmat(file_path)
        state_for_norm = np.concatenate((state_for_norm, data[state_var_name]), axis=1)

    # Calculate and store normalization params
    _, params_state = normalize_data(state_for_norm)
    print("Normalization parameters calculated.")

    # Load test files
    test_files = sorted(list(test_path.glob('*.mat')))
    test_data_raw = []
    for file_path in test_files:
        data = loadmat(file_path)
        test_data_raw.append({
            'control': data[control_var_name],
            'state': data[state_var_name]
        })

    # ==========================================================================
    # 3. Parameter Iteration Loop
    # ==========================================================================
    print("\n## 3. Starting parameter iteration... ##")

    for delay_step in delay_steps:
        for phi_dim in phi_dimensions:
            print(f"\n=== Training with delay_step={delay_step}, PhiDimensions={phi_dim} ===")

            # --- Neural Network Parameters ---
            params = {}
            params['state_size'] = 6
            params['delay_step'] = delay_step
            params['control_size'] = 6
            params['PhiDimensions'] = phi_dim
            params['hidden_size'] = int((params['PhiDimensions'] + params['state_size']) / 2)
            params['output_size'] = params['PhiDimensions'] - params['state_size']
            params['initialLearnRate'] = 0.001
            params['minLearnRate'] = 1e-6
            params['num_epochs'] = 1000
            params['L1'] = 500
            params['L2'] = 1.0
            params['L3'] = 0.0001
            params['batchSize'] = 512
            params['patience'] = 20
            params['lrReduceFactor'] = 0.2
            params['pred_step'] = loss_pred_step

            # Create model save directory for this parameter combination
            model_save_path = base_model_save_path / f"delay_{delay_step}_phi_{phi_dim}"
            model_save_path.mkdir(parents=True, exist_ok=True)

            # ==========================================================================
            # 4. Generate Training Data with current delay_step
            # ==========================================================================
            control_train_list, state_train_list, label_train_list = [], [], []
            for file_path in train_files:
                data = loadmat(file_path)
                state_data = data[state_var_name]
                if is_norm:
                    state_data, _ = normalize_data(state_data, params_state)

                ctrl_td, state_td, label_td = generate_lstm_data(
                    data[control_var_name], state_data, params['delay_step'], loss_pred_step
                )
                control_train_list.append(ctrl_td)
                state_train_list.append(state_td)
                label_train_list.append(label_td)

            # Concatenate data from all files
            control_train = np.concatenate(control_train_list, axis=2)
            state_train = np.concatenate(state_train_list, axis=1)
            label_train = np.concatenate(label_train_list, axis=2)

            train_data = {
                'control': control_train,
                'state': state_train,
                'label': label_train
            }

            # ==========================================================================
            # 5. Preprocess Test Data with current delay_step
            # ==========================================================================
            test_data = []
            for test_set in test_data_raw:
                control_test = test_set['control']
                state_test = test_set['state']
                if is_norm:
                    state_test, _ = normalize_data(state_test, params_state)

                test_data.append({
                    'control': control_test,
                    'state': state_test
                })

            # ==========================================================================
            # 6. Train the Network
            # ==========================================================================
            print(f"Training network with delay_step={delay_step}, PhiDimensions={phi_dim}...")
            net = train_lstm_lko(params, train_data, test_data)

            # Save the trained model with parameter-specific name
            model_filename = model_save_path / f'model_delay_{delay_step}_phi_{phi_dim}.pth'
            torch.save(net.state_dict(), model_filename)
            print(f"Model saved to '{model_filename}'")

            # ==========================================================================
            # 7. Final Evaluation
            # ==========================================================================
            print("Evaluating model...")
            final_rmse_scores = []

            for i, test_set in enumerate(test_data):
                control_test_raw = test_set['control']
                state_test_raw = test_set['state']

                # Prepare data for evaluation
                pred_step_eval = control_test_raw.shape[1] - params['delay_step']
                control_test_raw = control_test_raw[:, 0:pred_step_eval]
                initial_state_eval = torch.from_numpy(state_test_raw[:, :params['delay_step']]).unsqueeze(0)
                initial_state_eval = initial_state_eval.permute(0, 2, 1).float()

                control_eval = torch.from_numpy(control_test_raw).permute(1, 0).float()

                true_labels_eval = np.zeros([params['state_size'], pred_step_eval])
                for k in range(pred_step_eval):
                    true_labels_eval[:, k] = state_test_raw[:, k + params['delay_step']]

                true_labels_eval = torch.from_numpy(true_labels_eval)

                # Evaluate
                *_, y_true, y_pred = evaluate_lstm_lko(net, control_eval, initial_state_eval, true_labels_eval,
                                                       params['delay_step'])

                # Denormalize if needed
                if is_norm:
                    y_pred = denormalize_data(y_pred, params_state)
                    y_true = denormalize_data(y_true, params_state)

                rmse_score = np.sqrt(np.mean((y_true - y_pred) ** 2))
                final_rmse_scores.append(rmse_score)

            mean_final_rmse = np.mean(final_rmse_scores)
            print(f'--- Average RMSE: {mean_final_rmse:.4f} ---')

            # Store results
            results.append({
                'delay_step': delay_step,
                'phi_dimensions': phi_dim,
                'rmse': mean_final_rmse,
                'model_path': str(model_filename)
            })

    # ==========================================================================
    # 8. Save and Sort Results
    # ==========================================================================
    print("\n## 8. Saving and sorting results... ##")
    # Convert results to DataFrame and sort by RMSE
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values(by='rmse')

    # Save results to CSV with UTF-8 encoding
    results_csv_path = results_save_path / "parameter_tuning_results.csv"
    results_df.to_csv(results_csv_path, index=False, encoding="utf-8")
    print(f"Results saved to '{results_csv_path}'")

    # Save full results to text file
    text_path = results_save_path / "parameter_tuning_results.txt"
    with open(text_path, "w", encoding="utf-8") as f:
        f.write("所有参数组合按RMSE排序 (最佳到最差):\n")
        f.write(results_df.to_string())

    # Print all parameter combinations sorted by RMSE
    print("\n所有参数组合按RMSE排序 (最佳到最差):")
    formatted_df = results_df[['delay_step', 'phi_dimensions', 'rmse']].copy()

    # 设置列格式化选项（左对齐文本，右对齐数字）
    formatted_df['delay_step'] = formatted_df['delay_step'].astype(str).str.rjust(10)
    formatted_df['phi_dimensions'] = formatted_df['phi_dimensions'].astype(str).str.rjust(15)
    formatted_df['rmse'] = formatted_df['rmse'].apply(lambda x: f"{x:>15.6f}")

    # 创建对齐的列名
    headers = [
        str('delay_step').rjust(10),
        str('phi_dimensions').rjust(15),
        str('rmse').rjust(15)
    ]

    # 打印表头和分隔线
    print(" ".join(headers))
    print("-" * (10 + 15 + 15 + 2))  # 列宽总和加间距

    # 打印对齐的数据行
    for _, row in formatted_df.iterrows():
        print(f"{row['delay_step']} {row['phi_dimensions']} {row['rmse']}")

    # Save top model using shutil
    best_result = results_df.iloc[0]
    best_model_path = Path(best_result['model_path'])
    top_model_save_path = base_model_save_path / "top_model.pth"
    shutil.copy(best_model_path, top_model_save_path)
    print(f"\nBest model copied to '{top_model_save_path}'")

    # 可选：将完整结果保存为文本文件
    with open(results_save_path / "parameter_tuning_results.txt", "w") as f:
        f.write(str(results_df))


if __name__ == '__main__':
    main()