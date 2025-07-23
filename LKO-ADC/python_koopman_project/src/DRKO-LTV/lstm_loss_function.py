import torch
import torch.nn.functional as F


def lstm_loss_function(model, state, control, label, L1, L2, L3):
    # 1. 从输入张量的形状中获取维度信息
    device = state.device
    batch_size = state.shape[0]
    state_size = state.shape[2]
    delay_step = state.shape[1]  # history window size
    pred_step = control.shape[1]

    # 初始化列表，用于存储每个预测步骤的张量
    phi_pred_list = []
    phi_true_list = []
    state_pred_list = []
    state_true_list = []
    state_next_list = []
    state_decode_list = []

    # 2. 迭代地进行预测并收集结果
    # 循环的第一次迭代使用原始输入状态

    state_sequence = state

    for i in range(pred_step):
        # 当前步升维以及预测
        control_current = control[:, i, -1, :]
        control_history_sequence = control[:, i, :-1, :]
        state_current = state_sequence[:, -1, :]
        state_true = state_current
        state_history_sequence = state_sequence[:, :-1, :]
        phi_current, phi_pred, state_pred = model(state_current, control_current, state_history_sequence,
                                                  control_history_sequence)
        state_decode = model.C(phi_current)

        # 获取标签的真实值
        control_history_sequence_next = control[:, i, 1:, :]
        state_next = label[:, i, -1, :]
        state_history_sequence_next = label[:, i, :-1, :]
        phi_next, *_ = model(state_next, control_current, state_history_sequence_next, control_history_sequence_next)

        # 将当前步的结果添加到列表中
        phi_pred_list.append(phi_pred)
        phi_true_list.append(phi_next)
        state_pred_list.append(state_pred)
        state_true_list.append(state_true)
        state_next_list.append(state_next)
        state_decode_list.append(state_decode)

        # 更新状态
        state_sequence = torch.cat((state_sequence[:, 1:, :], state_pred.unsqueeze(1)), dim=1)

    # 3. 将列表中的张量拼接成一个大张量以便计算损失
    # cat(..., dim=0) 会将 (batch, features) 的列表变成 (batch * pred_step, features)
    all_phi_pred = torch.cat(phi_pred_list, dim=0)
    all_phi_true = torch.cat(phi_true_list, dim=0)
    all_state_pred = torch.cat(state_pred_list, dim=0)
    all_state_true = torch.cat(state_true_list, dim=0)
    all_state_next = torch.cat(state_next_list, dim=0)
    all_state_decode = torch.cat(state_decode_list, dim=0)

    # 4. 计算各个部分的损失
    # 使用均方误差损失 (MSE)，它与RMSE在优化上是等价的
    loss_state = F.mse_loss(all_state_pred, all_state_next)
    loss_phi = F.mse_loss(all_phi_pred, all_phi_true)
    loss_decode = F.mse_loss(all_state_true, all_state_decode)

    # 5. 计算加权的最终总损失
    total_loss = (L1 * loss_state) + (L2 * loss_phi) + (L3 * loss_decode)

    return total_loss


def lstm_loss_function2(model, state, control, label, L1, L2, L3):
    """
    一个强化的多步损失函数，包含对整个预测时域的线性一致性约束。

    此版本旨在通过惩罚多步高维预测的累积误差，直接提升模型在纯线性
    多步预测中的表现，使其更适用于MPC。

    Args:
        model (nn.Module): 您的 LKO_lstm_Network 模型实例。
        state (Tensor): 初始历史状态序列, shape: (B, T, d)。
        control (Tensor): 控制输入序列, shape: (B, N, T, c)。
        label (Tensor): 真实未来状态序列, shape: (B, N, T, d)。
        L1 (float): 多步状态预测损失的权重 (Prediction Loss)。
        L2 (float): 多步线性一致性损失的权重 (Linearity Loss)。
        L3 (float): 解码重构损失的权重 (Reconstruction Loss)。

    Returns:
        total_loss (Tensor): 加权后的总损失。
    """
    # --- 1. 准备 predict_multistep_lifted 函数的输入 ---

    # 初始状态和历史 (t时刻)
    state_current = state[:, -1, :]
    state_history_sequence = state[:, :-1, :]

    # 与初始状态窗对应的控制历史
    control_history_sequence = control[:, 0, :-1, :]

    # 未来N步的控制输入序列, shape: (B, N, c)
    future_control_sequence = control[:, :, -1, :]

    # 未来N步的真实状态标签, shape: (B, N, d)
    future_label_sequence = label[:, :, -1, :]

    # 获取预测时域的长度 N
    predict_horizon = future_control_sequence.shape[1]

    # --- 2. 一次性执行多步高维预测 ---

    phi_current, phi_pred_seq, state_pred_seq = model.predict_multistep_lifted(
        state_current,
        state_history_sequence,
        control_history_sequence,  # 注意这里用了初始的控制历史
        future_control_sequence
    )
    # phi_pred_seq shape: (B, N, output_size)
    # state_pred_seq shape: (B, N, d)

    # --- 3. 计算各个部分的损失 ---

    # Loss 1: 多步状态预测损失 (Prediction Loss)
    # 衡量在原始空间中的预测准确性
    loss_pred = F.mse_loss(state_pred_seq, future_label_sequence)

    # Loss 2: 解码重构损失 (Reconstruction Loss)
    # 衡量初始状态的编解码效果
    state_reconstructed = model.C(phi_current)
    loss_decode = F.mse_loss(state_reconstructed, state_current)

    # Loss 3: 多步线性一致性损失 (Multi-step Linearity Loss) - **核心改进**
    # 目标: 让模型预测的高维序列 phi_pred_seq 尽可能接近真实状态编码后的高维序列 phi_true_seq

    phi_true_list = []
    # 循环遍历未来N步的每一个真实标签
    for i in range(predict_horizon):
        # 提取第 i 个未来时间点的真实状态及其对应的历史
        # i=0 时, 对应 t+1 时刻的状态和历史
        state_true_step_i = label[:, i, -1, :]
        state_history_step_i = label[:, i, :-1, :]

        # 提取导致该状态的控制输入 (u_{t+i})
        control_current_step_i = control[:, i, -1, :]

        # 提取该状态对应的控制历史 (u_{t+i-T+2}, ..., u_{t+i-1})
        # 注意: 根据您的数据生成方式，label[:,i,:,:] 对应的控制历史是 control[:,i,1:,:]
        control_history_step_i = control[:, i, 1:, :]

        # 调用模型编码器，得到该真实状态的高维表示
        phi_true_step, _, _ = model(
            state_true_step_i,
            control_current_step_i,
            state_history_step_i,
            control_history_step_i
        )
        phi_true_list.append(phi_true_step)

    # 将列表堆叠成一个序列张量, shape: (B, N, output_size)
    phi_true_seq = torch.stack(phi_true_list, dim=1)

    # 计算整个高维预测序列与真实编码序列之间的误差
    # 使用 .detach() 是标准做法，将真实编码序列作为固定目标，不通过它进行反向传播
    loss_linear = F.mse_loss(phi_pred_seq, phi_true_seq.detach())

    # --- 4. 计算加权的最终总损失 ---
    total_loss = (L1 * loss_pred) + (L2 * loss_linear) + (L3 * loss_decode)

    return total_loss
