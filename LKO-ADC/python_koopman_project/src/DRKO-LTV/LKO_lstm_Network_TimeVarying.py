import torch
import torch.nn as nn


class LKO_lstm_Network_TimeVarying(nn.Module):

    def __init__(self, state_size, hidden_size_lstm, hidden_size_mlp, output_size, control_size, time_step):
        """
        构造函数：定义网络的所有层。
        此版本实现了时变 (Time-Varying) 的 Koopman 算子 A 和 B。

        Args:
            state_size (int): 状态向量的特征维度 (d)。
            hidden_size_lstm (int): LSTM 隐藏层的大小。
            hidden_size_mlp (int): 编码器 MLP 的隐藏层大小。
            output_size (int): 'phi' 特征输出的维度 (升维后的维度)。
            control_size (int): 控制向量的特征维度 (c)。
            time_step (int): 输入状态序列的时间步长度 (m)。
        """
        super().__init__()
        self.output_size = output_size
        self.control_size = control_size

        # 1. 定义 baseLayers lstm，用于处理状态序列
        # 这个部分保持不变，它负责从历史数据中提取特征
        self.base_lstm = nn.Sequential(
            nn.LSTM(state_size + control_size, hidden_size_lstm, 1, batch_first=True)
        )

        # 编码器 MLP，用于将 (当前状态 + LSTM历史编码) 映射到高维空间
        self.base_mlp = nn.Sequential(
            nn.Linear(hidden_size_lstm + state_size, hidden_size_mlp),
            nn.ELU(),
            nn.Linear(hidden_size_mlp, output_size),
            nn.ELU(),
        )

        # 2. 【核心改动】定义时变算子 A 和 B 的生成器网络
        # 移除固定的 self.A 和 self.B
        # self.A = nn.Linear(output_size, output_size, bias=False)  <-- 被移除
        # self.B = nn.Linear(control_size, output_size, bias=False) <-- 被移除

        # A 是一个 (output_size x output_size) 的矩阵
        self.A_elements = output_size * output_size
        # B 是一个 (output_size x control_size) 的矩阵
        self.B_elements = output_size * control_size

        self.A = torch.zeros(self.output_size, self.output_size)
        self.B = torch.zeros(self.output_size, self.control_size)

        # 这个全连接层接收 LSTM 的最终隐藏状态，并生成 A 和 B 矩阵的所有元素
        self.operator_generator_fc = nn.Linear(hidden_size_lstm, self.A_elements + self.B_elements)

        # 3. 定义解码器 C，将高维特征映射回原始状态空间
        # 这部分保持不变
        self.C = nn.Sequential(
            nn.Linear(output_size, state_size),
        )

    def _get_time_varying_operators(self, last_hidden_state):
        """
        一个辅助函数，根据 LSTM 的输出动态生成时变算子 A_tv 和 B_tv。

        Args:
            last_hidden_state (torch.Tensor): LSTM 的最终隐藏状态，形状为 (B, hidden_size_lstm)。

        Returns:
            A_tv (torch.Tensor): 时变算子 A，形状为 (B, output_size, output_size)。
            B_tv (torch.Tensor): 时变算子 B，形状为 (B, output_size, control_size)。
        """
        batch_size = last_hidden_state.shape[0]

        # 生成 A 和 B 矩阵的扁平化向量
        # (B, hidden_size_lstm) -> (B, A_elements + B_elements)
        operator_params = self.operator_generator_fc(last_hidden_state)

        # 切分出 A 和 B 的部分
        A_flat = operator_params[:, :self.A_elements]
        B_flat = operator_params[:, self.A_elements:]

        # 将扁平化的向量重塑为矩阵
        # (B, A_elements) -> (B, output_size, output_size)
        A_tv = A_flat.view(batch_size, self.output_size, self.output_size)
        # (B, B_elements) -> (B, output_size, control_size)
        B_tv = B_flat.view(batch_size, self.output_size, self.control_size)
        self.A = A_tv
        self.B = B_tv

        return A_tv, B_tv

    def forward(self, state_current, control_current, state_sequence, control_sequence):
        # 1. LSTM 编码历史信息
        history_sequence = torch.cat((state_sequence, control_sequence), dim=2)
        _, (hn, _) = self.base_lstm(history_sequence)
        last_hidden_state = hn[-1, :, :]  # (B, hidden_size_lstm)

        # 2. MLP 编码得到当前高维状态 phi_current
        hidden_state = torch.cat([state_current, last_hidden_state], dim=1)
        phi_current = self.base_mlp(hidden_state)  # (B, output_size)

        # 3. 【核心改动】根据历史信息生成时变算子 A 和 B
        A_tv, B_tv = self._get_time_varying_operators(last_hidden_state)

        # 4. 在高维空间中进行一步预测
        # 使用批量矩阵乘法 (torch.bmm) 来应用时变算子
        # A * phi(t)
        # A_tv: (B, out, out), phi_current.unsqueeze(-1): (B, out, 1) -> (B, out, 1)
        out_A = torch.bmm(A_tv, phi_current.unsqueeze(-1)).squeeze(-1)

        # B * u(t)
        # B_tv: (B, out, ctrl), control_current.unsqueeze(-1): (B, ctrl, 1) -> (B, out, 1)
        out_B = torch.bmm(B_tv, control_current.unsqueeze(-1)).squeeze(-1)

        # phi_pred = A(t) * phi(t) + B(t) * u(t)
        phi_pred = out_A + out_B

        # 5. 解码回原始状态空间
        state_pred = self.C(phi_pred)

        # 返回时变算子 A 和 B 以便进行分析或施加约束
        return phi_current, phi_pred, state_pred

    def predict_multistep_lifted(self, state_current, state_sequence, control_sequence, future_control_sequence):
        # 1. 编码得到初始高维状态 phi_current
        history_sequence = torch.cat((state_sequence, control_sequence), dim=2)
        _, (hn, _) = self.base_lstm(history_sequence)
        last_hidden_state = hn[-1, :, :]

        hidden_state = torch.cat([state_current, last_hidden_state], dim=1)
        phi_current = self.base_mlp(hidden_state)

        # 2. 【核心改动】生成时变算子 A 和 B
        # 对于多步预测，我们基于给定的初始历史信息生成一组 A 和 B
        # 并在整个预测时域中使用这组固定的算子
        A_tv, B_tv = self._get_time_varying_operators(last_hidden_state)

        # 3. 在高维空间中进行多步闭环预测
        phi_pred_list = []
        state_pred_list = []

        predict_horizon = future_control_sequence.shape[1]
        g_t = phi_current  # g_t 是高维状态的迭代变量

        for i in range(predict_horizon):
            u_t = future_control_sequence[:, i, :]  # (B, control_size)

            # 核心步骤：在高维空间中进行一步线性预测
            # g(t+1) = A_tv * g(t) + B_tv * u(t)
            # 使用与 forward 中相同的批量矩阵乘法
            g_t_plus_1 = torch.bmm(A_tv, g_t.unsqueeze(-1)).squeeze(-1) + \
                         torch.bmm(B_tv, u_t.unsqueeze(-1)).squeeze(-1)

            # 将预测出的高维状态映射回原始状态空间
            state_t_plus_1 = self.C(g_t_plus_1)

            # 收集结果
            phi_pred_list.append(g_t_plus_1)
            state_pred_list.append(state_t_plus_1)

            # 更新高维状态用于下一次循环
            g_t = g_t_plus_1

        # 4. 后处理：将结果列表堆叠成张量
        phi_pred_sequence = torch.stack(phi_pred_list, dim=1)
        state_pred_sequence = torch.stack(state_pred_list, dim=1)

        return phi_current, phi_pred_sequence, state_pred_sequence

    def dimension_lift(self, state_current, state_sequence, control_sequence):
        # 此函数仅用于升维，不涉及 A, B 算子，因此保持不变
        history_sequence = torch.cat((state_sequence, control_sequence), dim=2)
        out, (hn, cn) = self.base_lstm(history_sequence)
        last_hidden_state = hn[-1, :, :]
        hidden_state = torch.cat([state_current, last_hidden_state], dim=1)
        phi_current = self.base_mlp(hidden_state)
        return phi_current