import torch

# 假设 denormalize_data 和 calculate_rmse 函数已在别处定义
# from your_utils import denormalize_data, calculate_rmse

# 文件名: evaluate_hybrid_model.py

import torch
import numpy as np

# 假设 denormalize_data 和 calculate_rmse 函数已在别处定义
from src.normalize_data import denormalize_data
from src.calculate_rmse import calculate_rmse


def evaluate_hybrid_ltv_model(model, initial_state_sequence, initial_control_sequence,
                              future_labels, future_controls, params_state, is_norm):
    """
    接收已切分好的数据，使用 HybridLTVKoopmanNetwork 模型进行多步预测评估。
    """
    # 1. 准备工作 (Setup)
    model.eval()
    device = next(model.parameters()).device

    predict_step = 100  # 或者可以从 future_controls.shape[0] 获取

    # 将所有数据移动到正确的设备，并增加批大小维度 (B=1)
    initial_state_sequence = initial_state_sequence.to(device).unsqueeze(0)
    initial_control_sequence = initial_control_sequence.to(device).unsqueeze(0)
    future_labels = future_labels.to(device).unsqueeze(0)
    future_controls = future_controls.to(device).unsqueeze(0)

    # print(initial_state_sequence.shape, initial_control_sequence.shape)

    # 2. 准备模型输入 (现在非常清晰)
    #    state_history 的 shape 变为 (1, delay_step-1, d), 是3D的
    state_current = initial_state_sequence[:, -1, :]
    state_history = initial_state_sequence[:, :-1, :]

    #    control_history 的 shape 变为 (1, delay_step-1, c), 也是3D的
    control_history = initial_control_sequence[:, :-1, :]

    #    未来的控制序列

    future_control_sequence = future_controls[:, :predict_step, -1, :]

    # 3. 执行多步预测
    with torch.no_grad():
        # 调用模型内置的多步预测方法
        # 这里的输入维度现在是匹配的
        _, _, _, state_pred_seq, _, _, _ = model.predict_multistep_lifted(
            state_current,
            state_history,
            control_history,
            future_control_sequence
        )

    # 4. 后处理 (Post-processing)
    y_pred = state_pred_seq.squeeze(0).t()

    # 从未来的真实标签中提取用于比较的部分
    y_true = future_labels.squeeze(0)[:predict_step, -1, :].t()

    if is_norm:
        y_pred = denormalize_data(y_pred, params_state)
        y_true = denormalize_data(y_true, params_state)

    rmse = calculate_rmse(y_pred, y_true)

    return rmse.item(), y_true.cpu().numpy(), y_pred.cpu().numpy()
