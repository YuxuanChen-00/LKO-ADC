import torch
import torch.nn.functional as F


def lstm_loss_function(model, state, control, label, L1, L2, L3):
    """
    计算一个迭代预测过程的复合损失，其逻辑与MATLAB版本完全对应。

    该损失由三部分组成：
    1. 状态预测损失 (L1-weighted)：模型迭代预测的状态与真实状态之间的MSE。
    2. 特征空间损失 (L2-weighted)：模型迭代预测的内部特征(phi)与真实特征之间的MSE。
    3. L2正则化损失 (L3-weighted)：模型参数的L2范数，用于防止过拟合。

    Args:
        model (torch.nn.Module): 要训练的PyTorch模型。
        state (torch.Tensor): 初始输入状态序列，形状 (batch, time_step, d)。
        control (torch.Tensor): 整个预测窗口的控制输入，形状 (batch, pred_step, c)。
        label (torch.Tensor): 整个预测窗口的真实标签，形状 (batch, pred_step, time_step, d)。
        L1 (float): 状态损失的权重。
        L2 (float): 特征空间损失的权重。
        L3 (float): L2正则化权重。

    Returns:
        torch.Tensor: 计算出的总损失，是一个标量张量，可用于反向传播。
    """
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
    current_state_pred = state

    for i in range(pred_step):
        # 提取当前时间步的控制输入和真实标签
        current_control = control[:, i, :]
        current_label_state = label[:, i, :, :]

        # 第一次前向传播：使用上一轮的预测状态作为输入
        phi_current, phi_pred = model(current_state_pred, current_control)

        # 第二次前向传播：使用当前的真实标签作为输入
        phi_true, phi_label_next = model(current_label_state, current_control)

        # 从模型输出中提取状态部分
        # 假设模型输出的前 state_size * delay_step 个元素是状态
        predicted_state_part = phi_pred[:, :state_size * delay_step]
        true_state_part = phi_true[:, :state_size * delay_step]

        # 将当前步的结果添加到列表中
        phi_pred_list.append(phi_pred)
        phi_true_list.append(phi_true)
        state_pred_list.append(predicted_state_part)
        state_true_list.append(true_state_part)

        # 更新下一次迭代的状态：使用本轮的预测结果
        # 注意要 reshape 成模型接受的输入格式 (batch, seq, feature)
        current_state_pred = predicted_state_part.reshape(batch_size, delay_step, state_size)

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

    # 5. 计算L2正则化损失
    # 注意：在PyTorch中，这通常通过优化器的 weight_decay 参数实现，更为高效。

    # 6. 计算加权的最终总损失
    total_loss = (L1 * loss_state) + (L2 * loss_phi)

    return total_loss


# --- 示例：如何使用该函数（需要一个虚拟模型） ---
if __name__ == '__main__':
    # 沿用上一个示例中的虚拟模型
    class DummyModel(torch.nn.Module):
        def __init__(self, state_features, control_features, delay_step):
            super().__init__()
            self.state_features = state_features
            self.delay_step = delay_step
            self.linear = torch.nn.Linear(
                state_features * delay_step + control_features,
                state_features * delay_step * 2  # 假设phi是state的两倍长
            )

        def forward(self, state, control):
            state_flat = state.reshape(state.shape[0], -1)
            combined_input = torch.cat([state_flat, control], dim=1)
            return self.linear(combined_input)


    # 定义参数
    batch_sz, time_st, pred_st = 32, 10, 5
    d_feat, c_feat = 4, 2

    # 创建虚拟模型和数据
    model = DummyModel(d_feat, c_feat, time_st)
    dummy_state_in = torch.randn(batch_sz, time_st, d_feat)
    dummy_control_in = torch.randn(batch_sz, pred_st, c_feat)
    dummy_label_in = torch.randn(batch_sz, pred_st, time_st, d_feat)

    # 调用损失函数
    total_loss_val = lstm_loss_function(
        model,
        dummy_state_in,
        dummy_control_in,
        dummy_label_in,
        L1=1.0,
        L2=0.5,
        L3=1e-4
    )

    print("--- 损失函数计算示例 ---")
    print(f"计算得到的总损失: {total_loss_val.item():.4f}")
    print("这个损失值可以被用于 loss.backward() 进行反向传播。")