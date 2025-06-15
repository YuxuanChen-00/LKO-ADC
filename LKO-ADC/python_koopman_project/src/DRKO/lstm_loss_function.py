import torch
import torch.nn.functional as F


def lstm_loss_function(model, state, control, label, L1, L2):
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

    # 2. 迭代地进行预测并收集结果
    # 循环的第一次迭代使用原始输入状态

    state_sequence = state

    for i in range(pred_step):
        # 当前步升维以及预测
        control_current = control[:, i, -1, :]
        control_history_sequence = control[:, i, :-1, :]
        state_current = state_sequence[:, -1, :]
        state_history_sequence = state_sequence[:, :-1, :]
        phi_current, phi_pred, state_pred = model(state_current, control_current, state_history_sequence,
                                                  control_history_sequence)

        # 获取标签的真实值
        control_history_sequence_next = control[:, i, 1:, :]
        state_true = label[:, i, -1, :]
        state_history_sequence_next = label[:, i, :-1, :]
        phi_true, *_ = model(state_true, control_current, state_history_sequence_next, control_history_sequence_next)

        # 将当前步的结果添加到列表中
        phi_pred_list.append(phi_pred)
        phi_true_list.append(phi_true)
        state_pred_list.append(state_pred)
        state_true_list.append(state_true)

        # 更新状态
        state_sequence = torch.cat((state_sequence[:, 1:, :], state_pred.unsqueeze(1)), dim=1)

    # 3. 将列表中的张量拼接成一个大张量以便计算损失
    # cat(..., dim=0) 会将 (batch, features) 的列表变成 (batch * pred_step, features)
    all_phi_pred = torch.cat(phi_pred_list, dim=0)
    all_phi_true = torch.cat(phi_true_list, dim=0)
    all_state_pred = torch.cat(state_pred_list, dim=0)
    all_state_true = torch.cat(state_true_list, dim=0)

    # 4. 计算各个部分的损失
    # 使用均方误差损失 (MSE)，它与RMSE在优化上是等价的
    loss_state = F.mse_loss(all_state_pred, all_state_true)
    loss_phi = F.mse_loss(all_phi_pred, all_phi_true)

    # 5. 计算加权的最终总损失
    total_loss = (L1 * loss_state) + (L2 * loss_phi)

    return total_loss


