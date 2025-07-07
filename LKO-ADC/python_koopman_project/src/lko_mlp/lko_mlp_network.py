# 文件名: lko_lstm_network.py
import torch
import torch.nn as nn


class LKO_mlp_Network(nn.Module):
    """
    MATLAB 的 lko_mlp_network 类的 PyTorch 等效实现。
    这个网络接收一个状态序列和一个控制向量作为输入，并通过一个复杂的图结构来计算输出。
    """

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

        # 1. 定义 baseLayers MLP，用于处理状态序列
        # 在PyTorch中，当nn.Linear接收 (B, T, C) 的输入时，
        # 它会自动地、独立地作用于每个时间步 T。
        self.base_mlp = nn.Sequential(
            nn.Linear(state_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size * 2),
            nn.Tanh(),
            nn.Linear(hidden_size * 2, hidden_size * 4),
            nn.Tanh(),
            nn.Linear(hidden_size * 4, hidden_size * 8),
            nn.Tanh(),
            nn.Linear(hidden_size * 8, hidden_size * 16),
            nn.Tanh(),
            nn.Linear(hidden_size * 16, hidden_size * 32),
            nn.Tanh(),
            nn.Linear(hidden_size * 32, hidden_size * 64),
            nn.Tanh(),
            nn.Linear(hidden_size * 64, hidden_size * 32),
            nn.Tanh(),
            nn.Linear(hidden_size * 32, hidden_size * 16),
            nn.Tanh(),
            nn.Linear(hidden_size * 16, hidden_size * 8),
            nn.Tanh(),
            nn.Linear(hidden_size * 8, hidden_size * 4),
            nn.Tanh(),
            nn.Linear(hidden_size * 4, hidden_size * 2),
            nn.Tanh(),
            nn.Linear(hidden_size * 2, output_size),
            nn.LeakyReLU(),
        )

        # 2. 定义额外的全连接层 A 和 B
        # 'A' 层的输入是拼接后的(state, phi)，'B'层的输入是control
        # 拼接后的特征维度
        concatenated_features = (state_size * time_step) + (output_size * time_step)
        # A 和 B 层的输出维度（用于相加）
        add_layer_output_features = (time_step * state_size) + (output_size * time_step)

        # 线性层 A，无偏置，对应 'A'
        self.A = nn.Linear(concatenated_features, add_layer_output_features, bias=False)

        # 线性层 B，无偏置，对应 'B'
        self.B = nn.Linear(control_size, add_layer_output_features, bias=False)

    def forward(self, state_input, control_input):
        """
        定义网络的前向传播逻辑。

        Args:
            state_input (torch.Tensor): 状态序列输入。
                形状: (batch_size, time_step, state_size)
            control_input (torch.Tensor): 控制向量输入。
                形状: (batch_size, control_size)

        Returns:
            torch.Tensor: 网络的最终输出。
        """
        # 1. 状态序列通过基础MLP网络，得到 phi 特征序列
        # 输入: (B, T, C_state) -> 输出: (B, T, C_output)
        phi_sequence = self.base_mlp(state_input)

        # 2. Reshape state_input (对应 MATLAB 中的 'reshape1')
        # (B, T, C_state) -> (B, T * C_state)
        batch_size, time_step, _ = state_input.shape
        reshaped_state = state_input.reshape(batch_size, -1)

        # 3. Reshape phi_sequence (对应 MATLAB 中的 'reshape2')
        # (B, T, C_output) -> (B, T * C_output)
        reshaped_phi = phi_sequence.reshape(batch_size, -1)

        # 4. 拼接 reshaped_state 和 reshaped_phi (对应 'concat')
        # 按特征维度 (dim=1) 拼接
        phi_current = torch.cat([reshaped_state, reshaped_phi], dim=1)

        # 5. 通过 A 和 B 层
        # out_A 对应 'A' 层的输出
        out_A = self.A(phi_current)
        # out_B 对应 'B' 层的输出
        out_B = self.B(control_input)

        # 6. 相加 (对应 'add')
        phi_pred = out_A + out_B

        return phi_current, phi_pred