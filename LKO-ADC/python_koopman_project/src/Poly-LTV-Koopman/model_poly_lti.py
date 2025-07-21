import numpy as np
from scipy.io import loadmat
from pathlib import Path
import itertools
import os
from poly_lift import polynomial_expansion_td
from dataloader import generate_koopman_data


def calculate_koopman_operator(U: np.ndarray, X_lifted: np.ndarray, Y_lifted: np.ndarray) -> tuple:
    """使用最小二乘法计算Koopman算子 A 和 B。"""
    Omega = np.vstack([X_lifted, U])
    G = Y_lifted @ np.linalg.pinv(Omega)
    g_dim = X_lifted.shape[0]
    A = G[:, :g_dim]
    B = G[:, g_dim:]
    return A, B


def predict_multistep_koopman(A: np.ndarray, B: np.ndarray, U_future: np.ndarray, psi_initial: np.ndarray,
                              horizon: int) -> np.ndarray:
    """使用Koopman算子进行多步预测。"""
    g_dim = A.shape[0]
    psi_pred = np.zeros((g_dim, horizon))
    psi_current = psi_initial
    for i in range(horizon):
        u_current = U_future[:, i].reshape(-1, 1)
        psi_next = A @ psi_current + B @ u_current
        psi_pred[:, i] = psi_next.flatten()
        psi_current = psi_next
    return psi_pred


def calculate_rmse(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    """计算均方根误差 (RMSE)"""
    return np.sqrt(np.mean((y_pred - y_true) ** 2))


# ==============================================================================
# 2. 主执行程序
# ==============================================================================
if __name__ == '__main__':

    # --- 参数设置 ---
    print("## 1. 设置参数... ##")
    IS_NORM = False
    DELAY_TIME = 1
    TARGET_DIMENSIONS = 12
    PRED_STEP = 10  # 对于DMD，我们关心一步预测，所以固定为1

    # --- 路径设置 ---
    current_dir = Path(__file__).resolve().parent
    base_data_path = current_dir.parent.parent / "data" / "SorotokiData" / "MotionData8" / "FilteredDataPos"
    train_path = base_data_path / "80minTrain"
    test_path = base_data_path / "50secTest"
    model_save_path = current_dir / "models" / "SorotokiPoly"
    model_save_path.mkdir(parents=True, exist_ok=True)

    control_var_name = 'input'
    state_var_name = 'state'

    # 预测窗口设置 (与您的MATLAB代码一致)
    predict_start_offset = 10
    predict_horizon = 100
    state_window_slice = slice(0, 6)  # 用于评估RMSE的状态维度

    # --- 训练阶段 ---
    print("\n## 2. 开始训练阶段... ##")
    train_files = sorted(list(train_path.glob('*.mat')))

    control_data_list, state_data_list, label_data_list = [], [], []

    for file_path in train_files:
        data = loadmat(file_path)
        # 调用您提供的函数来生成带窗口的数据
        control_win, state_win, label_win = generate_koopman_data(
            data[control_var_name], data[state_var_name], DELAY_TIME, PRED_STEP
        )
        control_data_list.append(control_win)
        state_data_list.append(state_win)
        label_data_list.append(label_win)

    # 将所有训练数据在样本维度上合并
    control_win_all = np.concatenate(control_data_list, axis=0)
    state_win_all = np.concatenate(state_data_list, axis=0)
    label_win_all = np.concatenate(label_data_list, axis=0)


    label_win_all =np.expand_dims(label_win_all[:, 0], axis=1)

    # --- 数据整形：将窗口数据转换为EDMD所需的2D矩阵 ---
    num_samples, _, state_dim = state_win_all.shape
    _, _, _, control_dim = control_win_all.shape

    # state_all: (state_dim * delay_time, num_samples)
    state_all = state_win_all.reshape(num_samples, -1).T
    # label_all: (state_dim * delay_time, num_samples)
    label_all = label_win_all.squeeze(axis=1).reshape(num_samples, -1).T
    # control_all: (control_dim, num_samples) -> 取每个窗口最后一个控制输入
    control_all = control_win_all[:, 0, -1, :].T

    print(f"训练数据加载和整形完成。总样本数: {state_all.shape[1]}")

    if IS_NORM:
        print("数据归一化... (功能未实现)")
        params_state, params_control = None, None

    # 升维
    print("对训练数据进行升维...")
    state_lifted = polynomial_expansion_td(state_all, TARGET_DIMENSIONS, DELAY_TIME)
    label_lifted = polynomial_expansion_td(label_all, TARGET_DIMENSIONS, DELAY_TIME)

    # 计算Koopman算子
    print("计算Koopman算子 A 和 B...")
    A, B = calculate_koopman_operator(control_all, state_lifted, label_lifted)

    # 保存模型
    model_filename = model_save_path / f'poly_delay{DELAY_TIME}_lift{TARGET_DIMENSIONS}.npz'
    np.savez(model_filename, A=A, B=B)
    print(f"模型已保存至: {model_filename}")

    # --- 测试阶段 ---
    print("\n## 3. 开始测试阶段... ##")
    test_files = sorted(list(test_path.glob('*.mat')))
    all_rmse = []

    for i, file_path in enumerate(test_files):
        data = loadmat(file_path)
        # 为单个测试文件生成窗口数据
        control_win_td, state_win_td, label_win_td = generate_koopman_data(
            data[control_var_name], data[state_var_name], DELAY_TIME, PRED_STEP
        )

        label_win_td = np.expand_dims(label_win_td[:, 0], axis=1)

        # --- 数据整形 ---
        num_test_samples, _, _ = state_win_td.shape
        state_td = state_win_td.reshape(num_test_samples, -1).T
        label_td = label_win_td.squeeze(axis=1).reshape(num_test_samples, -1).T
        control_td = control_win_td[:, 0, -1, :].T

        print(label_td.shape)

        if IS_NORM:
            pass

        # 升维
        state_td_phi = polynomial_expansion_td(state_td, TARGET_DIMENSIONS, DELAY_TIME)

        # 定义预测的起始点和窗口
        start_idx = predict_start_offset - DELAY_TIME
        predict_indices = slice(start_idx, start_idx + predict_horizon)

        # 准备预测输入
        psi_initial = state_td_phi[:, start_idx].reshape(-1, 1)
        U_future = control_td[:, predict_indices]

        # 执行多步预测
        Y_pred_lifted = predict_multistep_koopman(A, B, U_future, psi_initial, predict_horizon)

        # 提取用于比较的真实值和预测值
        Y_true = label_td[:, predict_indices]
        Y_pred = Y_pred_lifted[:Y_true.shape[0], :]

        if IS_NORM:
            pass

        # 只评估前6个状态量
        Y_pred_eval = Y_pred[state_window_slice, :]
        Y_true_eval = Y_true[state_window_slice, :]

        rmse = calculate_rmse(Y_pred_eval, Y_true_eval)
        all_rmse.append(rmse)
        print(f"  - 测试文件 {i + 1}/{len(test_files)} ({file_path.name}): RMSE = {rmse:.6f}")

    # --- 显示统计结果 ---
    print("\n================= 综合测试结果 =================")
    print(f"平均 RMSE: {np.mean(all_rmse):.6f}")
    print(f"标准差 (Std Dev): {np.std(all_rmse):.6f}")
    print(f"最小 RMSE: {np.min(all_rmse):.6f}")
    print(f"最大 RMSE: {np.max(all_rmse):.6f}")
    print("各测试案例RMSE:")
    print(np.array(all_rmse))