import torch
import torch.nn as nn


# 辅助模块 1: 历史编码器 (无改动)
class HistoryEncoder(nn.Module):
    def __init__(self, state_dim, control_dim, rnn_hidden_dim, num_rnn_layers=1):
        super().__init__()
        self.gru = nn.GRU(
            input_size=state_dim + control_dim,
            hidden_size=rnn_hidden_dim,
            num_layers=num_rnn_layers,
            batch_first=True
        )

    def forward(self, state_history, control_history):
        combined_history = torch.cat((state_history, control_history), dim=-1)
        _, hn = self.gru(combined_history)
        context_vector = hn[-1]
        return context_vector


# 辅助模块 2: 矩阵生成头 (无改动)
class MatrixGenerator(nn.Module):
    def __init__(self, context_dim, g_dim, u_dim, state_dim, mlp_hidden_dim):
        super().__init__()
        self.g_dim, self.u_dim, self.state_dim = g_dim, u_dim, state_dim
        self.num_A_elements = g_dim * g_dim
        self.num_B_elements = g_dim * u_dim
        self.num_C_elements = state_dim * g_dim
        total_output_elements = self.num_A_elements + self.num_B_elements + self.num_C_elements
        self.mlp = nn.Sequential(
            nn.Linear(context_dim, mlp_hidden_dim), nn.ELU(),
            nn.Linear(mlp_hidden_dim, mlp_hidden_dim), nn.ELU(),
            nn.Linear(mlp_hidden_dim, total_output_elements)
        )

    def forward(self, context_vector):
        flat_matrices = self.mlp(context_vector)
        flat_A = flat_matrices[:, :self.num_A_elements]
        flat_B = flat_matrices[:, self.num_A_elements: self.num_A_elements + self.num_B_elements]
        flat_C = flat_matrices[:, self.num_A_elements + self.num_B_elements:]
        delta_A_t = flat_A.view(-1, self.g_dim, self.g_dim)
        delta_B_t = flat_B.view(-1, self.g_dim, self.u_dim)
        delta_C_t = flat_C.view(-1, self.state_dim, self.g_dim)
        return delta_A_t, delta_B_t, delta_C_t


# ---------------------------------------------------------------------------

# 主网络模型: C矩阵不带偏置项
class HybridLTVKoopmanNetwork(nn.Module):
    def __init__(self, state_size, control_size, time_step,
                 g_dim, encoder_gru_hidden, encoder_mlp_hidden,
                 delta_rnn_hidden, delta_mlp_hidden):
        super().__init__()
        self.g_dim, self.state_size, self.control_size = g_dim, state_size, control_size

        # 1. 主编码器 (x -> g)
        self.base_gru = nn.GRU(state_size + control_size, encoder_gru_hidden, 1, batch_first=True)
        self.base_mlp = nn.Sequential(
            nn.Linear(encoder_gru_hidden + state_size, encoder_mlp_hidden), nn.ELU(),
            nn.Linear(encoder_mlp_hidden, g_dim), nn.ELU(),
        )

        # 2. 算子的时不变部分 (Static Part)
        self.A_static_layer = nn.Linear(g_dim, g_dim, bias=False)
        self.B_static_layer = nn.Linear(control_size, g_dim, bias=False)
        # <--- 修改点 1: 允许C层拥有可学习的偏置项 --->
        self.C_static_layer = nn.Linear(g_dim, state_size, bias=True)

        # 3. 算子的时变部分 (Time-Varying Part)
        self.delta_encoder = HistoryEncoder(state_size, control_size, delta_rnn_hidden)
        self.delta_generator = MatrixGenerator(delta_rnn_hidden, g_dim, control_size, state_size, delta_mlp_hidden)

    def forward(self, state_current, control_current, state_history, control_history):
        # 步骤 A: 编码 g(t)
        history_for_encoder = torch.cat((state_history, control_history), dim=-1)
        _, hn = self.base_gru(history_for_encoder)
        phi_current = self.base_mlp(torch.cat([state_current, hn[-1]], dim=1))

        # 步骤 B: 生成 delta_A, delta_B, delta_C
        context = self.delta_encoder(state_history, control_history)
        delta_A_t, delta_B_t, delta_C_t = self.delta_generator(context)

        # 步骤 C: 组合得到 A(t), B(t), C(t)
        A_t = self.A_static_layer.weight.unsqueeze(0) + delta_A_t
        B_t = self.B_static_layer.weight.unsqueeze(0) + delta_B_t
        C_t = self.C_static_layer.weight.unsqueeze(0) + delta_C_t
        # <--- 修改点 2: 获取C层的静态偏置 --->
        C_bias = self.C_static_layer.bias

        # 步骤 D: 一步预测
        term_A = torch.bmm(A_t, phi_current.unsqueeze(-1))
        term_B = torch.bmm(B_t, control_current.unsqueeze(-1))
        phi_pred = term_A.squeeze(-1) + term_B.squeeze(-1)

        # 步骤 E: 解码 (加入偏置项)
        # <--- 修改点 3: 在解码时加上偏置 --->
        state_pred = torch.bmm(C_t, phi_pred.unsqueeze(-1)).squeeze(-1) + C_bias
        state_decode = torch.bmm(C_t, phi_current.unsqueeze(-1)).squeeze(-1) + C_bias

        return phi_current, phi_pred, state_decode, state_pred, delta_A_t, delta_B_t, delta_C_t

    def predict_multistep_lifted(self, state_current, state_history, control_history, future_control_sequence):
        # 1. 编码初始状态 g(t)
        history_for_encoder = torch.cat((state_history, control_history), dim=-1)
        _, hn = self.base_gru(history_for_encoder)
        phi_current = self.base_mlp(torch.cat([state_current, hn[-1]], dim=1))

        # 2. 生成“冻结”的LTV算子
        context = self.delta_encoder(state_history, control_history)
        delta_A_t, delta_B_t, delta_C_t = self.delta_generator(context)
        A_frozen = self.A_static_layer.weight.unsqueeze(0) + delta_A_t
        B_frozen = self.B_static_layer.weight.unsqueeze(0) + delta_B_t
        C_frozen = self.C_static_layer.weight.unsqueeze(0) + delta_C_t
        # <--- 修改点 4: 获取C层的静态偏置 --->
        C_bias_frozen = self.C_static_layer.bias

        # 3. 多步预测循环
        phi_pred_list, state_pred_list = [], []
        predict_horizon = future_control_sequence.shape[1]
        g_t = phi_current

        for i in range(predict_horizon):
            u_t = future_control_sequence[:, i, :]
            g_t_plus_1 = torch.bmm(A_frozen, g_t.unsqueeze(-1)).squeeze(-1) + \
                         torch.bmm(B_frozen, u_t.unsqueeze(-1)).squeeze(-1)

            # <--- 修改点 5: 在多步预测的解码中加上偏置 --->
            state_t_plus_1 = torch.bmm(C_frozen, g_t_plus_1.unsqueeze(-1)).squeeze(-1) + C_bias_frozen

            phi_pred_list.append(g_t_plus_1)
            state_pred_list.append(state_t_plus_1)
            g_t = g_t_plus_1

        # 4. 整理并返回结果
        phi_pred_sequence = torch.stack(phi_pred_list, dim=1)
        state_pred_sequence = torch.stack(state_pred_list, dim=1)
        # <--- 修改点 6: 在初始状态的重构中加上偏置 --->
        state_decode = torch.bmm(C_frozen, phi_current.unsqueeze(-1)).squeeze(-1) + C_bias_frozen

        return phi_current, phi_pred_sequence, state_decode, state_pred_sequence, delta_A_t, delta_B_t, delta_C_t