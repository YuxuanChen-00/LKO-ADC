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
    """
    在闭环模式下评估lstm模型，模拟MATLAB的lko（leave-k-out）评估逻辑。

    Args:
        model (torch.nn.Module): 训练好的PyTorch模型。
        control_data (torch.Tensor): 序列化的控制输入，形状为 (c, predict_step)。
        initial_state (torch.Tensor): 用于启动预测的初始状态序列，
                                       形状为 (1, time_step, d)。
        true_labels (torch.Tensor): 用于比较的真实标签，形状为 (d, predict_step)。
        delay_step (int): 模型状态表示中的时间步数，与MATLAB版本中的delay_step对应。

    Returns:
        tuple[float, np.ndarray, np.ndarray]: 一个元组，包含:
            - rmse (float): 计算出的均方根误差。
            - y_true (np.ndarray): 真实标签数组。
            - y_pred (np.ndarray): 模型的预测结果数组。
    """
    # 1. 准备工作
    model.eval()  # 将模型设置为评估模式
    device = next(model.parameters()).device  # 获取模型所在的设备 (cpu/cuda)

    predict_step = control.shape[0]

    # 将所有数据移动到正确的设备
    state_sequence = initial_state_sequence.to(device)
    control = control[:, 0, :, :].to(device)
    label = label[:, 0, :, :].to(device)

    y_pred_list = []

    # 2. 执行闭环预测
    with torch.no_grad():  # 在此上下文中不计算梯度
        for i in range(predict_step):
            state_current = state_sequence[-1, :]
            state_history_sequence = state_sequence[:-1, :]
            current_control = control[i, -1, :]
            control_history_sequence = control[i, :-1, :]

            phi_current, phi_pred, state_pred = model(current_state, current_control, state_history_sequence, control_history_sequence)

            # 假设模型输出 phi_pred 的形状是 (1, output_features)
            # 移除批处理维度，方便处理
            phi_pred_flat = phi_pred.squeeze(0)

            y_pred_list.append(state_pred)

            # 更新状态，用于下一次迭代
            # a. 从模型输出中提取完整的下一状态
            next_state_flat = phi_pred_flat[:state_size * delay_step]
            # b. 将其重塑为模型需要的输入形状 (1, delay_step, state_size)
            current_state = next_state_flat.reshape(1, delay_step, state_size)

    # 3. 后处理
    # 将预测列表堆叠成一个张量 (predict_step, d)

    y_pred = torch.stack(y_pred_list, dim=0)

    y_true = label[:, -1
    , :]

    # 计算RMSE
    rmse = calculate_rmse(y_pred, y_true)

    # 返回普通数值和NumPy数组，与MATLAB的输出格式保持一致
    return rmse.item(), y_true.cpu().numpy(), y_pred.cpu().numpy()


# --- 示例：如何使用该函数 ---
if __name__ == '__main__':
    # 假设我们已经有了上一部分定义的 generate_lstm_data 函数
    from __main__ import generate_lstm_data


    # 1. 定义一个符合输入要求的虚拟PyTorch模型
    class DummyModel(torch.nn.Module):
        def __init__(self, state_features, control_features, delay_step):
            super().__init__()
            self.state_features = state_features
            self.delay_step = delay_step
            # 一个简单的线性层，模拟从(state+control)到新状态的映射
            # 输入维度：state_features * delay_step + control_features
            # 输出维度：state_features * delay_step (用于更新状态)
            self.linear = torch.nn.Linear(
                state_features * delay_step + control_features,
                state_features * delay_step
            )

        def forward(self, state, control):
            # state: (batch, seq, features), control: (batch, features)
            # 将state展平以便与control拼接
            state_flat = state.reshape(state.shape[0], -1)  # (batch, seq*features)
            combined_input = torch.cat([state_flat, control], dim=1)
            output = self.linear(combined_input)
            return output


    # 2. 配置并生成模拟数据
    c_features = 2
    d_features = 4
    total_time = 100
    history_step = 10  # 对应 initial_state 的序列长度
    predict_step = 50  # 我们要连续预测多少步
    delay_step_val = 10  # 这个值必须与模型的历史窗口大小一致

    # a. 生成原始时序数据
    dummy_control_series = np.random.rand(c_features, total_time)
    dummy_state_series = np.random.rand(d_features, total_time)

    # b. 使用上一部分定义的函数来创建窗口化数据
    # 这里我们只需要一个样本来启动评估，所以设置 pred_step=1 和 num_samples=1
    ctrl_gen, state_gen, label_gen = generate_lstm_data(
        dummy_control_series, dummy_state_series, time_step=history_step, pred_step=predict_step
    )

    # 3. 准备评估函数的输入
    # a. 实例化模型
    model = DummyModel(d_features, c_features, delay_step=delay_step_val)

    # b. 提取初始状态 (取第一个样本的state)
    # 原始 shape (d, num_samples, history), permute后 (num_samples, history, d)
    initial_state_np = np.transpose(state_gen[:, 0:1, :], (1, 2, 0))
    initial_state_tensor = torch.from_numpy(initial_state_np).float()
    print(f"初始状态形状 (initial_state): {initial_state_tensor.shape}")

    # c. 提取用于迭代的控制序列
    # 原始 shape (c, pred_step, num_samples), 我们取第一个样本的控制序列
    control_data_np = ctrl_gen[:, :, 0]
    control_data_tensor = torch.from_numpy(control_data_np).float()
    print(f"闭环控制序列形状 (control_data): {control_data_tensor.shape}")

    # d. 提取真实标签
    # 这是最关键的一步，模拟MATLAB中复杂的squeeze操作
    # 我们需要的是从初始状态之后，未来 `predict_step` 个时间点的真实状态
    start_index_for_true_labels = history_step
    end_index_for_true_labels = history_step + predict_step
    true_labels_np = dummy_state_series[:, start_index_for_true_labels:end_index_for_true_labels]
    true_labels_tensor = torch.from_numpy(true_labels_np).float()
    print(f"真实标签形状 (true_labels): {true_labels_tensor.shape}\n")

    # 4. 运行评估函数
    rmse_val, y_true_out, y_pred_out = evaluate_lstm_lko(
        model=model,
        control_data=control_data_tensor,
        initial_state=initial_state_tensor,
        true_labels=true_labels_tensor,
        delay_step=delay_step_val
    )

    print("--- 评估结果 ---")
    print(f"RMSE: {rmse_val:.4f}")
    print(f"真实值Y_true的形状: {y_true_out.shape}")
    print(f"预测值Y_pred的形状: {y_pred_out.shape}")
