import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR
import numpy as np
import copy  # 用于深度拷贝最佳模型
import matplotlib.pyplot as plt  # 导入 matplotlib 库

# 假设这些模块已正确定义和可用
from evalutate_lstm_lko import evaluate_lstm_lko, evaluate_lstm_lko2
from generate_lstm_data import generate_lstm_data
from lstm_loss_function import lstm_loss_function
from lko_lstm_network import LKO_lstm_Network


# ==============================================================================
def train_lstm_lko(params, train_data, test_data):
    """
    MATLAB函数 train_lstm_lko 的直接Python/PyTorch转换。

    Args:
        params (dict): 包含所有超参数的字典。
        train_data (dict): 包含 'control', 'state', 'label' NumPy数组的训练数据字典。
        test_data (list[dict]): 测试数据集的列表，每个元素是一个包含测试数据的字典。

    Returns:
        torch.nn.Module: 训练好的、性能最佳的模型。
    """
    # 1. 参数设置
    # 从params字典解包参数
    state_size = params['state_size']
    delay_step = params['delay_step']
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

    # 3. 训练数据准备
    # 从字典中获取 NumPy 数组
    control_train_np = train_data['control']
    state_train_np = train_data['state']
    label_train_np = train_data['label']

    # 将数据转换为PyTorch张量，并转换维度以匹配 (batch, ...) 格式
    # 注意：数据首先留在CPU上，在训练循环中按批次移动到GPU，以节省显存
    # state:  (num_samples, time_step, d)
    state_train = torch.from_numpy(state_train_np).float().to(device)
    # control: (num_samples, pred_step, c)
    control_train = torch.from_numpy(control_train_np).float().to(device)
    # label: (num_samples, pred_step, time_step, d)
    label_train = torch.from_numpy(label_train_np).float().to(device)

    num_samples = state_train.shape[0]


    # 4. 网络初始化
    net = LKO_lstm_Network(state_size, hidden_size_lstm, hidden_size_mlp, output_size, control_size, delay_step)
    net.to(device)

    # 5. 训练设置
    optimizer = optim.Adam(net.parameters(), lr=initialLearnRate, weight_decay=1e-1)
    scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=minLearnRate)  # 使用 params['minLearnRate']

    best_test_loss = float('inf')
    # best_train_loss = float('inf') # 这个变量不再用于学习率调度，可以移除或保留用于记录最佳训练损失
    wait_counter = 0
    best_net_state_dict = None  # 用于存储最佳模型的权重

    # 用于记录每个 Epoch 的损失值
    train_losses = []
    test_losses = []
    learning_rates = []

    # 6. 自定义训练循环
    print("开始训练...")
    for epoch in range(num_epochs):
        net.train()  # 设置为训练模式

        # 手动打乱数据索引，模拟 MATLAB 的 shuffle(ds_train)
        shuffled_indices = np.random.permutation(num_samples)

        epoch_train_loss = 0.0  # 记录当前 epoch 的训练损失总和
        num_batches_processed = 0  # 记录处理的批次数量

        # 手动实现 minibatch 循环
        for i in range(0, num_samples, batchSize):
            # 丢弃最后一个不足尺寸的批次
            if i + batchSize > num_samples:
                continue

            # 获取当前批次的索引和数据
            batch_indices = shuffled_indices[i: i + batchSize]
            control_batch = control_train[batch_indices]
            state_batch = state_train[batch_indices]
            label_batch = label_train[batch_indices]

            # 计算损失和梯度
            optimizer.zero_grad()
            total_loss = lstm_loss_function(net, state_batch, control_batch, label_batch, L1, L2)
            total_loss.backward()

            # 更新参数 (Adam优化器)
            optimizer.step()

            epoch_train_loss += total_loss.item()
            num_batches_processed += 1

        # 在每个 epoch 结束时更新学习率
        scheduler.step()
        learning_rates.append(optimizer.param_groups[0]['lr'])  # 记录当前 epoch 的学习率

        # 计算当前 epoch 的平均训练损失
        avg_train_loss = epoch_train_loss / num_batches_processed if num_batches_processed > 0 else 0.0
        train_losses.append(avg_train_loss)

        # 7. 评估
        net.eval()  # 设置为评估模式
        test_loss_list = []

        # test_data 是一个字典列表
        for test_set in test_data:
            control_test = test_set['control']
            state_test = test_set['state']
            label_test = test_set['label']
            initial_state_sequence = state_test[0, :, :]

            # 调用评估函数
            with torch.no_grad():  # 评估时禁用梯度计算
                test_loss, _, _ = evaluate_lstm_lko(net, control_test, initial_state_sequence, label_test)
            test_loss_list.append(test_loss)

        if len(test_loss_list) > 0:
            mean_test_loss = np.mean(test_loss_list)
        else:
            mean_test_loss = float('inf')  # 如果没有有效的测试集，则设为无穷大

        test_losses.append(mean_test_loss)  # 记录当前 epoch 的平均测试损失

        if mean_test_loss < best_test_loss:
            best_test_loss = mean_test_loss
            # 深度拷贝当前模型的权重，而不是复制引用
            best_net_state_dict = copy.deepcopy(net.state_dict())
            # 如果需要，可以在此处保存到文件: torch.save(net.state_dict(), 'best_net.pth')
            wait_counter = 0  # 重置等待计数器
        else:
            wait_counter += 1

        # 8. 早停
        if wait_counter >= patience:
            print(f"测试损失在 {patience} 个 epoch 内没有改善，提前停止训练。")
            break  # 提前结束训练循环

        current_lr_for_print = optimizer.param_groups[0]['lr']

        print(f'Epoch {epoch + 1}/{num_epochs} | 训练集当前损失: {avg_train_loss:.4f} | '
              f'测试集均方根误差: {mean_test_loss:.4f} | 学习率: {current_lr_for_print:.6f}')

    print('\n训练完成！')

    # 9. 加载最佳权重到新模型并返回
    if best_net_state_dict:
        print(f"返回在测试集上表现最佳的模型 (RMSE: {best_test_loss:.4f})")
        best_net = LKO_lstm_Network(state_size, hidden_size_lstm, hidden_size_mlp, output_size, control_size, delay_step)
        best_net.load_state_dict(best_net_state_dict)
        return best_net
    else:
        print("训练过程中未产生更佳模型，返回最终模型。")
        return net
