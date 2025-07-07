# 文件名: lko_lstm_network.py
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

def init_weights(model):
    """
    根据层类型和特定名称对LKO网络进行自定义初始化。
    """
    # 遍历模型的所有模块(层)
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            # 对Koopman算子A和B进行特殊处理
            if 'A' in name:
                print(f"Initializing Koopman Operator A ({name}) as Identity Matrix.")
                # A 矩阵必须是方阵
                if module.in_features == module.out_features:
                    nn.init.eye_(module.weight)
                else:
                    # 如果不是方阵，虽然不常见，但可以采用Xavier初始化
                    nn.init.xavier_uniform_(module.weight, gain=nn.init.calculate_gain('linear'))
            elif 'B' in name:
                print(f"Initializing Koopman Operator B ({name}) as Zero Matrix.")
                nn.init.zeros_(module.weight)
            else:
                # 对其他全连接层（在MLP和C中）使用Kaiming初始化
                # 因为激活函数是ELU
                print(f"Initializing Linear layer ({name}) with Kaiming Uniform.")
                nn.init.kaiming_uniform_(module.weight, a=0, mode='fan_in',
                                         nonlinearity='leaky_relu')  # ELU用leaky_relu的设置
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

        elif isinstance(module, nn.LSTM):
            print(f"Initializing LSTM layer ({name}).")
            for param_name, param in module.named_parameters():
                if 'weight_ih' in param_name:
                    # 输入到隐藏层的权重使用Xavier初始化
                    nn.init.xavier_uniform_(param)
                elif 'weight_hh' in param_name:
                    # 循环权重使用正交初始化
                    nn.init.orthogonal_(param)
                elif 'bias' in param_name:
                    # 所有偏置初始化为0
                    nn.init.zeros_(param)
                    # **可选技巧**: 将遗忘门的偏置初始化为1，有助于初始时更好地记住信息
                    # PyTorch中bias的顺序是 [b_ii, b_if, b_ig, b_io]
                    # 遗忘门是第二个，所以我们要设置1/4到1/2的元素
                    # hidden_size = param.size(0) // 4
                    # param.data[hidden_size:2*hidden_size].fill_(1.0)
