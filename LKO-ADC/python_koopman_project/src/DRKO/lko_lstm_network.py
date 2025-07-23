
import torch
import torch.nn as nn


class LKO_lstm_Network(nn.Module):

    def __init__(self, state_size, hidden_size_lstm, hidden_size_mlp, output_size, control_size, time_step):
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
            nn.LSTM(state_size + control_size, hidden_size_lstm, 1, batch_first=True)
        )

        self.base_mlp = nn.Sequential(
            nn.Linear(hidden_size_lstm+state_size, hidden_size_mlp),
            nn.ELU(),
            nn.Linear(hidden_size_mlp, output_size),
            nn.ELU(),
        )

        # 2. 定义额外的全连接层 A 和 B
        # 线性层 A，无偏置，对应 'A'
        self.A = nn.Linear(output_size, output_size, bias=False)

        # 线性层 B，无偏置，对应 'B'
        self.B = nn.Linear(control_size, output_size, bias=False)

        # 假设高维特征到原特征是一个近似线性映射
        self.C = nn.Sequential(
            nn.Linear(output_size, state_size),
        )

    def forward(self, state_current, control_current, state_sequence, control_sequence):
        history_sequence = torch.cat((state_sequence, control_sequence), dim=2)

        # 输入: (B, T, C_state) -> 输出: (B, T, C_output)
        out, (hn, cn) = self.base_lstm(history_sequence)
        last_hidden_state = hn[-1, :, :]

        # 按特征维度  拼接
        hidden_state = torch.cat([state_current, last_hidden_state], dim=1)
        phi_current = self.base_mlp(hidden_state)

        # out_A 对应 'A' 层的输出
        out_A = self.A(phi_current)
        # out_B 对应 'B' 层的输出
        out_B = self.B(control_current)

        phi_pred = out_A + out_B
        state_pred = self.C(phi_pred)

        return phi_current, phi_pred, state_pred

    def predict_multistep_lifted(self, state_current, state_sequence, control_sequence, future_control_sequence):

        # 1. 初始状态升维：复用模型前向传播中的编码逻辑
        # 这部分与您模型中 forward 函数的前半部分完全相同
        history_sequence = torch.cat((state_sequence, control_sequence), dim=2)
        out, (hn, cn) = self.base_lstm(history_sequence)
        last_hidden_state = hn[-1, :, :]
        hidden_state = torch.cat([state_current, last_hidden_state], dim=1)

        # 得到初始的升维状态 g(t)，这是多步预测的起点
        phi_current = self.base_mlp(hidden_state)

        # 2. 在高维空间中进行多步闭环预测
        phi_pred_list = []
        state_pred_list = []

        # 获取预测的步数 N
        predict_horizon = future_control_sequence.shape[1]

        # 将初始高维状态作为循环的第一个状态
        g_t = phi_current

        for i in range(predict_horizon):
            # 获取当前步的控制输入 u(t+i)
            u_t = future_control_sequence[:, i, :]

            # 核心步骤：在高维空间中进行一步线性预测
            # g(t+1) = A * g(t) + B * u(t)
            # 我们使用 model.A(g_t) 而不是矩阵乘法，因为这是调用nn.Module的标准方式
            g_t_plus_1 = self.A(g_t) + self.B(u_t)

            # 将预测出的高维状态映射回原始状态空间，用于记录和计算损失
            # y_pred(t+1) = C * g(t+1)
            state_t_plus_1 = self.C(g_t_plus_1)

            # 收集每一步的预测结果
            phi_pred_list.append(g_t_plus_1)
            state_pred_list.append(state_t_plus_1)

            # 更新高维状态，用于下一次循环
            # 注意：这里我们用 g_t_plus_1（高维）作为下一次迭代的输入
            g_t = g_t_plus_1

        # 3. 后处理：将结果列表堆叠成一个张量
        # (B, N, F) -> Batch, Sequence Length, Features
        phi_pred_sequence = torch.stack(phi_pred_list, dim=1)
        state_pred_sequence = torch.stack(state_pred_list, dim=1)

        return phi_current, phi_pred_sequence, state_pred_sequence

    def dimension_lift(self, state_current, state_sequence, control_sequence):
        history_sequence = torch.cat((state_sequence, control_sequence), dim=2)

        # 输入: (B, T, C_state) -> 输出: (B, T, C_output)
        out, (hn, cn) = self.base_lstm(history_sequence)
        last_hidden_state = hn[-1, :, :]

        # 按特征维度  拼接
        hidden_state = torch.cat([state_current, last_hidden_state], dim=1)
        phi_current = self.base_mlp(hidden_state)
        return phi_current
