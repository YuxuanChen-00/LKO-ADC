import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader  # 导入 DataLoader
import numpy as np
import copy
import matplotlib.pyplot as plt

# 假设这些模块已正确定义和可用
from evaluate_lstm_lko import evaluate_lstm_lko, evaluate_lstm_lko2
from generate_lstm_data import generate_lstm_data
from lstm_loss_function import lstm_loss_function, lstm_loss_function2
from lko_lstm_network import LKO_lstm_Network
from lstm_dataloader import CustomTimeSeriesDataset


# 假设 CustomTimeSeriesDataset 类已在上面定义

# ==============================================================================
def train_lstm_lko(params, train_data, test_data):
    """
    使用 DataLoader 优化的 train_lstm_lko 函数。

    Args:
        params (dict): 包含所有超参数的字典。
        train_data (dict): 包含 'control', 'state', 'label' NumPy数组的训练数据字典。
        test_data (list[dict]): 测试数据集的列表。

    Returns:
        torch.nn.Module: 训练好的、性能最佳的模型。
    """
    # 1. 参数设置 (与原版相同)
    state_size = params['state_size']
    delay_step = params['delay_step']
    params_state = params['params_state']
    params_control = params['params_control']
    is_norm = params['is_norm']
    control_size = params['control_size']
    hidden_size_lstm = params['hidden_size_lstm']
    hidden_size_mlp = params['hidden_size_mlp']
    output_size = params['output_size']
    initialLearnRate = params['initialLearnRate']
    minLearnRate = params['minLearnRate']
    num_epochs = params['num_epochs']
    L1, L2, L3 = params['L1'], params['L2'], params['L3']
    batchSize = params['batchSize']
    patience = params['patience']
    lrReduceFactor = params['lrReduceFactor']
    device = params['device']

    # 3. 数据准备 (使用 DataLoader)
    # 创建 Dataset 对象
    train_dataset = CustomTimeSeriesDataset(train_data)

    # 创建 DataLoader 对象
    # num_workers > 0 会启用多进程数据加载，在Unix-like系统(Linux, macOS)上能显著提速
    # 在Windows上或Jupyter Notebook中，有时 num_workers 设为 0 更稳定
    # drop_last=True 确保所有批次大小一致，避免最后一个小批次对训练产生干扰
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batchSize,
        shuffle=True,  # 在每个 epoch 自动打乱数据
        num_workers=16,  # 根据你的CPU核心数调整，例如 4 或 8
        pin_memory=True,  # 如果GPU可用，可以加速CPU到GPU的数据传输
        drop_last=True  # 丢弃最后一个不完整的批次
    )

    # 4. 网络初始化 (与原版相同)
    net = LKO_lstm_Network(state_size, hidden_size_lstm, hidden_size_mlp, output_size, control_size, delay_step)
    net.to(device)

    # 5. 训练设置 (与原版相同)
    optimizer = optim.Adam(net.parameters(), lr=initialLearnRate)
    scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=minLearnRate)

    best_test_loss = float('inf')
    wait_counter = 0
    best_net_state_dict = None

    train_losses = []
    test_losses = []
    learning_rates = []

    # 6. 自定义训练循环 (使用 DataLoader)
    print("开始使用 DataLoader 进行训练...")
    for epoch in range(num_epochs):
        net.train()
        epoch_train_loss = 0.0

        # DataLoader 使得训练循环非常简洁
        # 它会自动处理批次划分、数据打乱等操作
        for state_batch, control_batch, label_batch in train_loader:
            # 将当前批次的数据移动到目标设备 (GPU/CPU)
            state_batch = state_batch.to(device)
            control_batch = control_batch.to(device)
            label_batch = label_batch.to(device)

            # 计算损失和梯度
            optimizer.zero_grad()
            total_loss = lstm_loss_function(net, state_batch, control_batch, label_batch, L1, L2)
            total_loss.backward()
            optimizer.step()

            epoch_train_loss += total_loss.item()

        # 更新学习率
        scheduler.step()
        learning_rates.append(optimizer.param_groups[0]['lr'])

        # 计算平均训练损失
        # len(train_loader) 直接给出总批次数
        avg_train_loss = epoch_train_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        # 7. 评估 (与原版相同)
        if epoch % 1 == 0:
            net.eval()
            test_loss_list = []
            for test_set in test_data:
                control_test = test_set['control']
                state_test = test_set['state']
                label_test = test_set['label']
                initial_state_sequence = state_test[10 - delay_step, :, :]

                with torch.no_grad():
                    test_loss, _, _ = evaluate_lstm_lko(net, control_test[10 - delay_step:], initial_state_sequence,
                                                        label_test[10 - delay_step:], params_state, is_norm)
                test_loss_list.append(test_loss)

            mean_test_loss = np.mean(test_loss_list) if test_loss_list else float('inf')
            test_losses.append(mean_test_loss)

            if mean_test_loss < best_test_loss:
                best_test_loss = mean_test_loss
                best_net_state_dict = copy.deepcopy(net.state_dict())
                wait_counter = 0
            else:
                wait_counter += 1

        # # 8. 早停 (与原版相同)
        # if wait_counter >= patience:
        #     print(f"测试损失在 {patience} 个 epoch 内没有改善，提前停止训练。")
        #     break

        print(f'Epoch {epoch + 1}/{num_epochs} | 训练集当前损失: {avg_train_loss:.4f} | '
              f'测试集均方根误差: {mean_test_loss:.4f} | 学习率: {scheduler.get_last_lr()[0]:.6f}')

    print('\n训练完成！')

    # 9. 加载最佳模型并返回 (与原版相同)
    if best_net_state_dict:
        print(f"返回在测试集上表现最佳的模型 (RMSE: {best_test_loss:.4f})")
        best_net = LKO_lstm_Network(state_size, hidden_size_lstm, hidden_size_mlp, output_size, control_size,
                                    delay_step)
        best_net.load_state_dict(best_net_state_dict)
        return best_net
    else:
        print("训练过程中未产生更佳模型，返回最终模型。")
        return net
