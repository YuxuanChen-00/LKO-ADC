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
