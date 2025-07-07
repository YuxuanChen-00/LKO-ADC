import numpy as np
import torch


def generate_mlp_data(control, states, time_step, pred_step=1):
    """
    将时间序列数据转换为适用于mlp训练的窗口化数据，其逻辑与MATLAB脚本完全对应。

    Args:
        control (np.ndarray): 控制输入数据，形状为 (c, t)，其中 c 是控制特征数，t 是时间步总数。
        states (np.ndarray): 状态数据，形状为 (d, t)，其中 d 是状态特征数，t 是时间步总数。
        time_step (int): 用作输入特征的过去时间步数量（即历史窗口大小 m）。
        pred_step (int, optional): 需要预测的未来时间步数量。默认为 1。

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray]: 一个包含三个NumPy数组的元组:
            - file_control (np.ndarray): 用于预测窗口的控制数据。
                形状: (c, pred_step, num_samples)
            - file_state (np.ndarray): 输入的状态序列 (历史数据)。
                形状: (d, num_samples, time_step)
            - file_label (np.ndarray): 标签状态序列 (未来数据)。
                形状: (d, pred_step, num_samples, time_step)

    关于维度的说明:
        输出的维度结构刻意保持了与原MATLAB脚本的一致性。
        在PyTorch中，您通常需要调整这些维度以匹配 (batch, sequence, features) 的标准格式。
        例如:
        `# 将形状从 (d, num_samples, time_step) 转换为 (num_samples, time_step, d)`
        `file_state_torch = torch.from_numpy(file_state).permute(1, 2, 0)`
    """
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
    # file_control: (特征, 预测步数, 样本数)
    # file_state:   (特征, 样本数, 历史步数)
    # file_label:   (特征, 预测步数, 样本数, 历史步数) -> 注意：这是一个不寻常的形状
    file_control = np.zeros((c, pred_step, num_samples))
    file_state = np.zeros((d, num_samples, time_step))
    file_label = np.zeros((d, pred_step, num_samples, time_step))

    # 通过在时间序列上滑动窗口来生成每个样本
    for i in range(num_samples):
        # 定义输入状态历史的索引窗口。
        # 这里进行反转以匹配MATLAB中 `end:-1:start` 的逻辑。
        history_indices = np.arange(i + time_step - 1, i - 1, -1)
        file_state[:, i, :] = states[:, history_indices]

        for k_idx in range(pred_step):
            k = k_idx + 1  # 使用1-based的k来计算标签的偏移量

            # 标签是状态窗口在未来偏移k步的结果
            label_indices = history_indices + k
            file_label[:, k_idx, i, :] = states[:, label_indices]

            # 控制输入对应于历史窗口的末端
            # MATLAB索引: sample_idx(i+1) + time_step + k - 2
            # Python索引: i + time_step + k_idx
            control_idx = i + time_step + k_idx
            file_control[:, k_idx, i] = control[:, control_idx]

    return file_control, file_state, file_label

def generate_mlp_data_prevcontrol(control, states, time_step, pred_step=1):
    """
    将时间序列数据转换为适用于mlp训练的窗口化数据，其逻辑与MATLAB脚本完全对应。
    增加了将过去时刻的控制输入也拼接到时间延迟的状态变量后面的功能。

    Args:
        control (np.ndarray): 控制输入数据，形状为 (c, t)，其中 c 是控制特征数，t 是时间步总数。
        states (np.ndarray): 状态数据，形状为 (d, t)，其中 d 是状态特征数，t 是时间步总数。
        time_step (int): 用作输入特征的过去时间步数量（即历史窗口大小 m）。
        pred_step (int, optional): 需要预测的未来时间步数量。默认为 1。

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray]: 一个包含三个NumPy数组的元组:
            - file_control_future (np.ndarray): 用于预测窗口的未来控制数据。
                形状: (c, pred_step, num_samples)
            - file_state_and_control_history (np.ndarray): 输入的拼接序列 (历史状态 + 历史控制)。
                形状: (d + c, num_samples, time_step)
            - file_label (np.ndarray): 标签状态序列 (未来状态数据)。
                形状: (d, pred_step, num_samples, time_step)

    关于维度的说明:
        输出的维度结构刻意保持了与原MATLAB脚本以及原始函数的一致性。
        在PyTorch中，您通常需要调整这些维度以匹配 (batch, sequence, features) 的标准格式。
        例如:
        `# 将形状从 (d + c, num_samples, time_step) 转换为 (num_samples, time_step, d + c)`
        `combined_history_tensor = torch.from_numpy(file_state_and_control_history).permute(1, 2, 0)`
    """
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
    # file_control_future: (控制特征, 预测步数, 样本数) - 保持不变
    # file_state_and_control_history: (状态特征 + 控制特征, 样本数, 历史步数) - 这是关键变化
    # file_label:   (状态特征, 预测步数, 样本数, 历史步数) - 保持不变
    file_control_future = np.zeros((c, pred_step, num_samples))
    file_state_and_control_history = np.zeros((d + c, num_samples, time_step)) # 维度 d+c
    file_label = np.zeros((d, pred_step, num_samples, time_step))

    # 通过在时间序列上滑动窗口来生成每个样本
    for i in range(num_samples):
        # 定义输入历史的索引窗口。
        # 这里进行反转以匹配MATLAB中 `end:-1:start` 的逻辑。
        history_indices = np.arange(i + time_step - 1, i - 1, -1)

        # 提取过去状态和过去控制，并沿第一个维度（特征维度）拼接
        # states[:, history_indices] 的形状是 (d, time_step)
        # control[:, history_indices] 的形状是 (c, time_step)
        # 拼接后，形状将是 (d + c, time_step)
        combined_history = np.vstack((states[:, history_indices], control[:, history_indices]))
        file_state_and_control_history[:, i, :] = combined_history

        for k_idx in range(pred_step):
            k = k_idx + 1  # 使用1-based的k来计算标签的偏移量

            # 标签是状态窗口在未来偏移k步的结果
            label_indices = history_indices + k
            file_label[:, k_idx, i, :] = states[:, label_indices]

            # 控制输入对应于历史窗口的末端到预测窗口的结束
            control_idx = i + time_step + k_idx
            file_control_future[:, k_idx, i] = control[:, control_idx]

    # 返回值名称也相应修改，以反映其内容
    return file_control_future, file_state_and_control_history, file_label

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
        ctrl_out, state_out, label_out = generate_mlp_data(
            dummy_control, dummy_states, time_step=m_history, pred_step=p_future
        )

        print("--- 输出NumPy数组形状 (与MATLAB一致) ---")
        print(f"file_control shape: {ctrl_out.shape}")
        print(f"file_state shape:   {state_out.shape}")
        print(f"file_label shape:   {label_out.shape}\n")

        # 4. 转换为PyTorch张量并调整维度
        # PyTorch中mlp的通用格式是 (batch, sequence, features)
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