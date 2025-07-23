import torch
import time
import numpy as np
from src.normalize_data import normalize_data, denormalize_data


# 假设之前的函数保存在了名为 data_utils 的文件中
# from data_utils import generate_lstm_data

# --- 辅助函数：计算RMSE ---
def calculate_rmse(y_pred, y_true):
    """
    使用PyTorch计算均方根误差 (RMSE)。

    Args:
        y_pred (torch.Tensor): 模型的预测值。
        y_true (torch.Tensor): 真实值。

    Returns:
        torch.Tensor: 一个包含RMSE值的标量张量。
    """
    # return np.sqrt(((y_pred - y_true) ** 2).mean())
    return torch.sqrt(((y_pred - y_true) ** 2).mean())


# --- 核心评估函数 ---
def evaluate_lstm_lko(model, control, initial_state_sequence, label, params_state, is_norm):

    # 1. 准备工作
    model.eval()  # 将模型设置为评估模式
    device = next(model.parameters()).device  # 获取模型所在的设备 (cpu/cuda)

    predict_step = control.shape[0]
    predict_step = 100

    # 将所有数据移动到正确的设备
    state_sequence = initial_state_sequence
    control = control[:, 0, :, :]
    label = label[:, 0, :, :]

    y_pred_list = []

    loop_times = []  # 存储每次循环耗时
    total_start = time.perf_counter()  # 总开始时间

    # 2. 执行闭环预测
    with torch.no_grad():  # 在此上下文中不计算梯度
        for i in range(predict_step):
            iter_start = time.perf_counter()  # 单次循环开始时间
            # 加载当前的变量
            state_current = state_sequence[-1, :].unsqueeze(0)
            state_history_sequence = state_sequence[:-1, :].unsqueeze(0)
            control_current = control[i, -1, :].unsqueeze(0)
            control_history_sequence = control[i, :-1, :].unsqueeze(0)

            # 模型预测
            phi_current, phi_pred, state_pred = model(state_current, control_current, state_history_sequence,
                                                      control_history_sequence)
            y_pred_list.append(state_pred.squeeze(0))

            # 更新状态，用于下一次迭代
            state_sequence = torch.cat((state_sequence[1:, :], state_pred), dim=0)

            # 记录本次循环耗时
            iter_time = time.perf_counter() - iter_start
            loop_times.append(iter_time)
            # print(f"Iter {i}: {iter_time:.6f} seconds")

    total_time = time.perf_counter() - total_start
    # print(f"Total time: {total_time:.6f} sec, Avg: {total_time / predict_step:.6f} sec/iter")

    # 3. 后处理
    # 将预测列表堆叠成一个张量 (predict_step, d)

    y_pred = torch.stack(y_pred_list, dim=0).t()
    y_true = label[0:predict_step, -1, :].t()

    if is_norm:
        y_pred = denormalize_data(y_pred, params_state)
        y_true = denormalize_data(y_true, params_state)

    # 计算RMSE
    rmse = calculate_rmse(y_pred, y_true)

    # 返回普通数值和NumPy数组，与MATLAB的输出格式保持一致
    return rmse.item(), y_true.cpu().numpy(), y_pred.cpu().numpy()


def evaluate_lstm_lko2(model, control, initial_state_sequence, label, params_state, is_norm):
    """
    使用Koopman算子在高维空间进行多步预测。

    该方法首先将初始状态提升到高维空间，然后完全在高维线性空间中
    进行多步预测，只在每一步将结果映射回原始空间用于记录。
    """
    # 1. 准备工作 (Setup)
    model.eval()
    device = next(model.parameters()).device

    predict_step = 20

    control = control[:, 0, :, :].to(device)
    label = label[:, 0, :, :].to(device)
    initial_state_sequence = initial_state_sequence.to(device)

    # ==================== 代码修改处 ====================
    # A 和 B 仍然是线性层，直接取 .weight
    koopman_A = model.A.weight
    koopman_B = model.B.weight

    # C 是一个 Sequential 容器，我们需要其最后一个线性层的权重
    koopman_C = model.C[-1].weight
    # ===================================================

    y_pred_list = []

    # 2. 初始状态升维 (Initial Lifting)
    with torch.no_grad():
        state_current = initial_state_sequence[-1, :].unsqueeze(0)
        state_history = initial_state_sequence[:-1, :].unsqueeze(0)

        control_current = control[0, -1, :].unsqueeze(0)
        control_history = control[0, :-1, :].unsqueeze(0)

        g_current, _, _ = model(state_current, control_current, state_history, control_history)

    # 3. 在高维空间中进行多步闭环预测
    with torch.no_grad():
        for i in range(predict_step):
            u_current = control[i, -1, :].unsqueeze(0)

            # 使用权重张量进行线性预测
            g_next = g_current @ koopman_A.T + u_current @ koopman_B.T

            # 使用解码矩阵C的权重进行映射
            y_pred = g_next @ koopman_C.T
            y_pred_list.append(y_pred.squeeze(0))

            g_current = g_next

    # 4. 后处理 (Post-processing)
    y_pred = torch.stack(y_pred_list, dim=0).t()
    y_true = label[0:predict_step, -1, :].t()

    if is_norm:
        y_pred = denormalize_data(y_pred, params_state)
        y_true = denormalize_data(y_true, params_state)

    rmse = calculate_rmse(y_pred, y_true)

    return rmse.item(), y_true.cpu().numpy(), y_pred.cpu().numpy()

