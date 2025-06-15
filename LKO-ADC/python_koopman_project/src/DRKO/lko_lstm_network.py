# 文件名: lko_lstm_network.py
import torch
import torch.nn as nn


class LKO_lstm_Network(nn.Module):

    def __init__(self, state_size, hidden_size, output_size, control_size, time_step):
        """
        构造函数：定义网络的所有层。

        Args:
            state_size (int): 状态向量的特征维度 (d)。
            hidden_size (int): 隐藏层的大小。
            output_size (int): 'phi' 特征输出的维度。
            control_size (int): 控制向量的特征维度 (c)。
            time_step (int): 输入状态序列的时间步长度 (m)。
        """
        super().__init__()

        # 1. 定义 baseLayers lstm，用于处理状态序列
        # 在PyTorch中，当nn.Linear接收 (B, T, C) 的输入时，
        # 它会自动地、独立地作用于每个时间步 T。
        self.base_lstm = nn.Sequential(
            nn.LSTM(state_size + control_size, hidden_size, 1, batch_first=True)
        )

        self.base_mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ELU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ELU(),
            nn.Linear(hidden_size, output_size),
        )

        # 2. 定义额外的全连接层 A 和 B
        # 'A' 层的输入是拼接后的(state, phi)，'B'层的输入是control
        # 拼接后的特征维度
        concatenated_features = state_size + output_size
        # A 和 B 层的输出维度（用于相加）
        add_layer_output_features = (time_step * state_size) + (output_size * time_step)

        # 线性层 A，无偏置，对应 'A'
        self.A = nn.Linear(concatenated_features, add_layer_output_features, bias=False)

        # 线性层 B，无偏置，对应 'B'
        self.B = nn.Linear(control_size, add_layer_output_features, bias=False)

        # 假设高维特征到原特征是一个近似线性映射
        self.C = nn.Linear(state_size, add_layer_output_features, bias=False)

    def forward(self, state_current, control_current, state_sequence, control_sequence):
        history_sequence = torch.cat((state_sequence, control_sequence), dim=2)

        # 输入: (B, T, C_state) -> 输出: (B, T, C_output)
        out, (hn, cn) = self.base_lstm(history_sequence)
        last_hidden_state = hn[-1, :, :]

        # 按特征维度  拼接
        hidden_state = torch.cat([state_current, last_hidden_state], dim=1)
        phi_current = self.base_mlp(self.C(hidden_state))

        # out_A 对应 'A' 层的输出
        out_A = self.A(phi_current)
        # out_B 对应 'B' 层的输出
        out_B = self.B(control_current)

        phi_pred = out_A + out_B
        state_pred = self.C(phi_pred)

        return phi_current, phi_pred, state_pred
