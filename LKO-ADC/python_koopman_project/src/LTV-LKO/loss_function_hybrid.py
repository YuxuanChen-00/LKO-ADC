import torch
import torch.nn.functional as F


# 假设您的模型和 predict_multistep_lifted 函数已定义
# from your_model_file import HybridLTVKoopmanNetwork

def hybrid_ltv_loss(model, state, control, label, L1, L2, L3, L_delta):
    """
    为 HybridLTVKoopmanNetwork 设计的、包含多步约束和delta矩阵正则化的损失函数。

    Args:
        model (nn.Module): 您的 HybridLTVKoopmanNetwork 模型实例。
        state (Tensor): 初始历史状态序列, shape: (B, T, d)。
        control (Tensor): 控制输入序列, shape: (B, N, T, c)。
        label (Tensor): 真实未来状态序列, shape: (B, N, T, d)。
        L1 (float): 多步状态预测损失的权重 (Prediction Loss)。
        L2 (float): 多步线性一致性损失的权重 (Linearity Loss)。
        L3 (float): 解码重构损失的权重 (Reconstruction Loss)。
        L_delta (float): delta矩阵范数的正则化权重 (Smoothness Regularization)。

    Returns:
        total_loss (Tensor): 加权后的总损失。
    """
    # --- 1. 准备输入数据 (与之前相同) ---

    state_current = state[:, -1, :]
    state_history_sequence = state[:, :-1, :]
    control_history_sequence = control[:, 0, :-1, :]
    future_control_sequence = control[:, :, -1, :]
    future_label_sequence = label[:, :, -1, :]
    predict_horizon = future_control_sequence.shape[1]

    # --- 2. 一次性执行多步高维预测 ---
    # <--- 已修改: 接收新增的 delta_A_t, delta_B_t 返回值 --->
    phi_current, phi_pred_seq, state_decode, state_pred_seq, delta_A_t, delta_B_t, delta_C_t = model.predict_multistep_lifted(
        state_current,
        state_history_sequence,
        control_history_sequence,
        future_control_sequence
    )

    # --- 3. 计算原有部分的损失 ---

    # Loss 1: 多步状态预测损失 (无变化)
    loss_pred = F.mse_loss(state_pred_seq, future_label_sequence)

    # Loss 2: 解码重构损失 (无变化)
    loss_decode = F.mse_loss(state_decode, state_current)

    # Loss 3: 多步线性一致性损失 (循环内部调用有微小改动)
    phi_true_list = []
    for i in range(predict_horizon):
        state_true_step_i = label[:, i, -1, :]
        state_history_step_i = label[:, i, :-1, :]
        control_current_step_i = control[:, i, -1, :]
        control_history_step_i = control[:, i, 1:, :]

        # <--- 已修改: model.forward 现在返回5个值，正确解包 --- >
        # 调用模型编码器，得到该真实状态的高维表示
        phi_true_step, _, _, _, _, _, _ = model(
            state_true_step_i,
            control_current_step_i,
            state_history_step_i,
            control_history_step_i
        )
        phi_true_list.append(phi_true_step)

    phi_true_seq = torch.stack(phi_true_list, dim=1)
    loss_linear = F.mse_loss(phi_pred_seq, phi_true_seq.detach())

    # --- 4. 新增损失项：Delta矩阵正则化 ---
    # 这是利用新模型结构的关键，用于控制平滑性
    loss_delta_norm = torch.mean(delta_A_t ** 2) + torch.mean(delta_B_t ** 2) + torch.mean(delta_C_t ** 2)

    # --- 5. 计算加权的最终总损失 ---
    total_loss = (L1 * loss_pred) + (L2 * loss_linear) + (L3 * loss_decode) + (L_delta * loss_delta_norm)

    return total_loss