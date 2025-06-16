import torch
import numpy as np


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
    return torch.sqrt(torch.mean((y_pred - y_true) ** 2))


# --- 核心评估函数 ---
def evaluate_lstm_lko(model, control, initial_state_sequence, label):

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

    # 2. 执行闭环预测
    with torch.no_grad():  # 在此上下文中不计算梯度
        for i in range(predict_step):
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

    # 3. 后处理
    # 将预测列表堆叠成一个张量 (predict_step, d)

    y_pred = torch.stack(y_pred_list, dim=0)

    y_true = label[0:predict_step, -1, :]

    # 计算RMSE
    rmse = calculate_rmse(y_pred, y_true)


    # 返回普通数值和NumPy数组，与MATLAB的输出格式保持一致
    return rmse.item(), y_true.cpu().numpy(), y_pred.cpu().numpy()



def evaluate_lstm_lko2(model, control, initial_state_sequence, label):

    # 1. 准备工作
    model.eval()  # 将模型设置为评估模式
    device = next(model.parameters()).device  # 获取模型所在的设备 (cpu/cuda)

    predict_step = control.shape[0]

    # 初始化数据
    state_sequence = initial_state_sequence
    control = control[:, 0, :, :]
    label = label[:, 0, :, :]

    state_current = state_sequence[-1, :].unsqueeze(0)
    state_history_sequence = state_sequence[:-1, :].unsqueeze(0)
    control_current = control[0, -1, :].unsqueeze(0)
    control_history_sequence = control[0, :-1, :].unsqueeze(0)
    phi_current, *_ = model(state_current, control_current, state_history_sequence,
                                              control_history_sequence)

    # 初始化预测序列
    y_pred_list = []

    # 加载模型中的参数
    layer_A = model.A
    layer_B = model.B
    layer_C = model.C


    # 2. 执行闭环预测
    with torch.no_grad():  # 在此上下文中不计算梯度
        for i in range(predict_step):

            # 加载控制输入
            control_current = control[i, -1, :].unsqueeze(0)

            # 预测
            phi_pred = layer_A(phi_current) + layer_B(control_current)
            state_pred = layer_C(phi_pred)
            y_pred_list.append(state_pred.squeeze(0))

            # 更新状态，用于下一次迭代
            phi_current = phi_pred

    # 3. 后处理
    # 将预测列表堆叠成一个张量 (predict_step, d)

    y_pred = torch.stack(y_pred_list, dim=0)

    y_true = label[:, -1, :]

    # 计算RMSE
    rmse = calculate_rmse(y_pred, y_true)


    # 返回普通数值和NumPy数组，与MATLAB的输出格式保持一致
    return rmse.item(), y_true.cpu().numpy(), y_pred.cpu().numpy()

