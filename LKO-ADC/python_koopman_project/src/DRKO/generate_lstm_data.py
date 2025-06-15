import numpy as np
import torch


def generate_lstm_data(control, states, time_step, pred_step=1):
    # 确保输入是numpy数组
    control = np.asarray(control)
    states = np.asarray(states)

    # 提取维度并进行验证
    c, t = control.shape
    d, t_check = states.shape

    if t != t_check:
        raise ValueError(f"时间维度不匹配: control有 {t} 个时间步, states有 {t_check} 个时间步。")

    # 计算可以生成的样本窗口数量
    # 需要足够的数据来容纳一个历史窗口(time_step)和一个预测窗口(pred_step)
    num_samples = t - time_step - pred_step + 1

    if num_samples < 1:
        raise ValueError(f"时间序列太短，无法满足指定的 time_step ({time_step}) 和 pred_step ({pred_step})。"
                         f"至少需要 {time_step + pred_step} 个数据点。")

    # 预分配NumPy数组以存储结果
    # 形状格式与MATLAB脚本保持一致:
    # file_control: (特征, 预测步数, 样本数, 历史步数)
    # file_state:   (特征, 样本数, 历史步数)
    # file_label:   (特征, 预测步数, 样本数, 历史步数) -> 注意：这是一个不寻常的形状
    file_control = np.zeros((c, pred_step, num_samples, time_step))
    file_state = np.zeros((d, num_samples, time_step))
    file_label = np.zeros((d, pred_step, num_samples, time_step))

    # 通过在时间序列上滑动窗口来生成每个样本
    for i in range(num_samples):
        # 定义输入状态历史的索引窗口。
        history_indices = np.arange(i, i + time_step, 1)
        file_state[:, i, :] = states[:, history_indices]

        for k_idx in range(pred_step):
            k = k_idx + 1  # 使用1-based的k来计算标签的偏移量

            # 标签是状态窗口在未来偏移k步的结果
            label_indices = history_indices + k
            file_label[:, k_idx, i, :] = states[:, label_indices]

            # 控制输入对应于历史窗口的末端
            file_control[:, k_idx, i, :] = control[:, history_indices]

    file_state = np.transpose(file_state, (1, 2, 0))
    file_control = np.transpose(file_control, (2, 1, 3, 0))
    file_label = np.transpose(file_label, (2, 1, 3, 0))

    return file_control, file_state, file_label


# --- 示例：如何使用该函数 ---
if __name__ == '__main__':
    # 1. 配置参数
    c_features = 2  # 控制特征的数量
    d_features = 4  # 状态特征的数量
    total_time = 50  # 数据中的总时间步
    m_history = 10  # time_step (历史窗口大小)
    p_future = 5  # pred_step (预测窗口大小)

    # 2. 生成模拟数据 (形状为: 特征, 时间)
    dummy_control = np.random.rand(c_features, total_time)
    dummy_states = np.random.rand(d_features, total_time)

    print("--- 输入数据形状 ---")
    print(f"控制数据形状: {dummy_control.shape}")
    print(f"状态数据形状: {dummy_states.shape}")
    print(f"历史窗口 (time_step): {m_history}")
    print(f"预测窗口 (pred_step): {p_future}\n")

    # 3. 运行转换函数
    try:
        ctrl_out, state_out, label_out = generate_lstm_data(
            dummy_control, dummy_states, time_step=m_history, pred_step=p_future
        )

        print("--- 输出NumPy数组形状 (与MATLAB一致) ---")
        print(f"file_control shape: {ctrl_out.shape}")
        print(f"file_state shape:   {state_out.shape}")
        print(f"file_label shape:   {label_out.shape}\n")

        # 4. 转换为PyTorch张量并调整维度
        # PyTorch中lstm的通用格式是 (batch, sequence, features)
        # 我们的 batch 维度是 num_samples (在ctrl_out中是第3维, 在state_out中是第2维)

        # 将 batch 维度（即样本数）移动到第一位
        state_tensor = torch.from_numpy(state_out).permute(1, 2, 0).float()
        control_tensor = torch.from_numpy(ctrl_out).permute(2, 1, 0).float()

        # label是4D的，比较特殊。我们按逻辑进行转换
        label_tensor = torch.from_numpy(label_out).permute(2, 1, 3, 0).float()

        print("--- 用于PyTorch的张量形状 ---")
        print(f"状态张量 (state_tensor) shape:  (batch, seq_len, features) = {state_tensor.shape}")
        print(f"控制张量 (control_tensor) shape:(batch, seq_len, features) = {control_tensor.shape}")
        # label_tensor 的形状含义是 (样本数, 预测步数, 历史窗口, 特征数)
        print(f"标签张量 (label_tensor) shape:  (batch, pred_steps, history_win, features) = {label_tensor.shape}")

    except ValueError as e:
        print(f"发生错误: {e}")