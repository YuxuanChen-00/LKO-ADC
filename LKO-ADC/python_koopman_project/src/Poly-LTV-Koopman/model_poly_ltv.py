import torch
import torch.nn as nn
import torch.nn.functional as F



def residual_koopman_loss(model,
                          phi_current, control_current,
                          state_history, control_history,
                          label_state,
                          L1, L_delta):
    """
    为 ResidualKoopmanNetwork 设计的损失函数。

    Args:
        model (nn.Module): ResidualKoopmanNetwork 模型实例。
        phi_current (Tensor): 当前时刻的、已经升维的状态, shape: (B, g)。
        control_current (Tensor): 当前时刻的控制输入, shape: (B, c)。
        state_history (Tensor): 历史状态序列, shape: (B, T, d)。
        control_history (Tensor): 历史控制序列, shape: (B, T, c)。
        label_state (Tensor): 真实的下一时刻原始状态, shape: (B, d)。
        L1 (float): 状态预测损失的权重 (Prediction Loss)。
        L_delta (float): delta矩阵范数的正则化权重 (Regularization)。

    Returns:
        total_loss (Tensor): 加权后的总损失。
    """
    # --- 1. 通过模型进行一步预测 ---
    # 模型现在返回4个值，我们需要全部接收
    state_pred, _, delta_A_t, delta_B_t = model(
        phi_current,
        control_current,
        state_history,
        control_history
    )

    # --- 2. 计算状态预测损失 (Prediction Loss) ---
    # 直接在原始状态空间中比较预测值和真实值
    loss_pred = F.mse_loss(state_pred, label_state)

    # --- 3. 计算Delta矩阵正则化损失 (Regularization) ---
    # 这是约束神经网络行为、防止过拟合的关键
    # 注意: 我们只正则化模型学习的部分，即 delta_A 和 delta_B
    loss_delta_norm = torch.mean(delta_A_t ** 2) + torch.mean(delta_B_t ** 2)

    # --- 4. 计算加权的最终总损失 ---
    total_loss = (L1 * loss_pred) + (L_delta * loss_delta_norm)

    return total_loss

class HistoryEncoder(nn.Module):
    """
    历史编码器，用于将状态和控制的历史序列压缩成一个上下文向量。
    (此模块与您提供的代码完全相同)
    """

    def __init__(self, state_dim, control_dim, rnn_hidden_dim, num_rnn_layers=1):
        super().__init__()
        self.gru = nn.GRU(
            input_size=state_dim + control_dim,
            hidden_size=rnn_hidden_dim,
            num_layers=num_rnn_layers,
            batch_first=True
        )

    def forward(self, state_history: torch.Tensor, control_history: torch.Tensor) -> torch.Tensor:
        # 确保输入是3D的: (batch, seq_len, features)
        combined_history = torch.cat((state_history, control_history), dim=-1)
        _, hn = self.gru(combined_history)
        context_vector = hn[-1]  # 取最后一层的隐藏状态
        return context_vector


class DeltaABGenerator(nn.Module):
    """
    一个专门用于生成残差矩阵 delta_A 和 delta_B 的神经网络模块。
    """

    def __init__(self, context_dim: int, g_dim: int, u_dim: int, mlp_hidden_dim: int):
        super().__init__()
        self.g_dim = g_dim
        self.u_dim = u_dim

        # 计算A和B矩阵的元素总数
        self.num_A_elements = g_dim * g_dim
        self.num_B_elements = g_dim * u_dim
        total_output_elements = self.num_A_elements + self.num_B_elements

        # 定义输出矩阵的MLP
        self.mlp = nn.Sequential(
            nn.Linear(context_dim, mlp_hidden_dim),
            nn.ELU(),
            nn.Linear(mlp_hidden_dim, mlp_hidden_dim),
            nn.ELU(),
            nn.Linear(mlp_hidden_dim, total_output_elements)
        )

    def forward(self, context_vector: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        flat_matrices = self.mlp(context_vector)

        # 分割并重塑成矩阵
        flat_A = flat_matrices[:, :self.num_A_elements]
        flat_B = flat_matrices[:, self.num_A_elements:]

        delta_A_t = flat_A.view(-1, self.g_dim, self.g_dim)
        delta_B_t = flat_B.view(-1, self.g_dim, self.u_dim)

        return delta_A_t, delta_B_t


# ===================================================================
# 主网络模型: ResidualKoopmanNetwork
# ===================================================================

class ResidualKoopmanNetwork(nn.Module):
    """
    一个接收预先计算的静态(A, B, C)矩阵，并只学习时变残差(delta_A, delta_B)的网络。
    """

    def __init__(self,
                 A_static: torch.Tensor,
                 B_static: torch.Tensor,
                 C_static: torch.Tensor,
                 state_dim: int,
                 control_dim: int,
                 delta_rnn_hidden: int,
                 delta_mlp_hidden: int):
        """
        初始化残差Koopman网络。

        Args:
            A_static (torch.Tensor): 预计算的静态A矩阵, 形状 (g_dim, g_dim)。
            B_static (torch.Tensor): 预计算的静态B矩阵, 形状 (g_dim, control_dim)。
            C_static (torch.Tensor): 预计算的静态C矩阵, 形状 (state_dim, g_dim)。
            state_dim (int): 原始状态维度。
            control_dim (int): 控制输入维度。
            delta_rnn_hidden (int): 历史编码器的隐藏层维度。
            delta_mlp_hidden (int): 残差矩阵生成器的MLP隐藏层维度。
        """
        super().__init__()

        g_dim = A_static.shape[0]
        self.state_dim = state_dim
        self.control_dim = control_dim
        self.g_dim = g_dim

        # 将静态矩阵注册为buffer，它们是模型的一部分但不是可训练参数
        self.register_buffer('A_static', A_static)
        self.register_buffer('B_static', B_static)
        self.register_buffer('C_static', C_static)

        # 唯一的可训练部分：用于生成残差的网络
        self.delta_encoder = HistoryEncoder(state_dim, control_dim, delta_rnn_hidden)
        self.delta_generator = DeltaABGenerator(delta_rnn_hidden, g_dim, control_dim, delta_mlp_hidden)

    def forward(self,
                phi_current: torch.Tensor,
                control_current: torch.Tensor,
                state_history: torch.Tensor,
                control_history: torch.Tensor) -> tuple:
        """
        执行一步前向预测。

        Args:
            phi_current (torch.Tensor): 当前时刻的升维状态, 形状 (batch, g_dim)。
            control_current (torch.Tensor): 当前时刻的控制输入, 形状 (batch, control_dim)。
            state_history (torch.Tensor): 历史状态序列, 形状 (batch, seq_len, state_dim)。
            control_history (torch.Tensor): 历史控制序列, 形状 (batch, seq_len, control_dim)。

        Returns:
            A tuple containing:
            - state_pred (torch.Tensor): 预测的下一时刻原始状态, 形状 (batch, state_dim)。
            - phi_pred (torch.Tensor): 预测的下一时刻升维状态, 形状 (batch, g_dim)。
        """
        # 1. 神经网络部分：生成时变残差
        context = self.delta_encoder(state_history, control_history)
        delta_A_t, delta_B_t = self.delta_generator(context)

        # 2. 组合得到时变算子 A(t) 和 B(t)
        # unsqueeze(0) 用于将静态矩阵广播到批次中的每个样本
        A_t = self.A_static.unsqueeze(0) + delta_A_t
        B_t = self.B_static.unsqueeze(0) + delta_B_t

        # 3. 在升维空间中进行一步线性预测
        # 使用bmm(batch matrix-matrix product)进行批量矩阵乘法
        phi_pred = torch.bmm(A_t, phi_current.unsqueeze(-1)).squeeze(-1) + \
                   torch.bmm(B_t, control_current.unsqueeze(-1)).squeeze(-1)

        # 4. 使用固定的C矩阵解码回原始状态空间
        state_pred = torch.bmm(self.C_static.unsqueeze(0).expand(phi_pred.shape[0], -1, -1),
                               phi_pred.unsqueeze(-1)).squeeze(-1)

        return state_pred, phi_pred, delta_A_t, delta_B_t

    def predict_multistep(self,
                          phi_initial: torch.Tensor,
                          future_control_sequence: torch.Tensor,
                          state_history: torch.Tensor,
                          control_history: torch.Tensor) -> tuple:
        """
        执行多步开环预测。

        Args:
            phi_initial (torch.Tensor): 初始的升维状态, 形状 (batch, g_dim)。
            future_control_sequence (torch.Tensor): 未来的控制输入序列, 形状 (batch, horizon, control_dim)。
            state_history (torch.Tensor): 用于生成算子的历史状态序列。
            control_history (torch.Tensor): 用于生成算子的历史控制序列。

        Returns:
            torch.Tensor: 预测的未来状态序列, 形状 (batch, horizon, state_dim)。
        """
        # 1. 生成"冻结"的LTV算子，它在整个预测时域内保持不变
        context = self.delta_encoder(state_history, control_history)
        delta_A_t, delta_B_t = self.delta_generator(context)
        A_frozen = self.A_static.unsqueeze(0) + delta_A_t
        B_frozen = self.B_static.unsqueeze(0) + delta_B_t

        # 2. 循环进行多步预测
        phi_t = phi_initial
        state_pred_list = []
        predict_horizon = future_control_sequence.shape[1]

        for i in range(predict_horizon):
            u_t = future_control_sequence[:, i, :]

            # 在升维空间中演化
            phi_t_plus_1 = torch.bmm(A_frozen, phi_t.unsqueeze(-1)).squeeze(-1) + \
                           torch.bmm(B_frozen, u_t.unsqueeze(-1)).squeeze(-1)

            # 解码得到当前步的预测状态
            state_t_plus_1 = torch.bmm(self.C_static.unsqueeze(0).expand(phi_t_plus_1.shape[0], -1, -1),
                                       phi_t_plus_1.unsqueeze(-1)).squeeze(-1)

            state_pred_list.append(state_t_plus_1)

            # 更新升维状态以进行下一步预测
            phi_t = phi_t_plus_1

        # 3. 将结果堆叠成一个张量
        state_pred_sequence = torch.stack(state_pred_list, dim=1)
        return state_pred_sequence, delta_A_t, delta_B_t
