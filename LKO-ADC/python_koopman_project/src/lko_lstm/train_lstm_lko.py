import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import copy  # 用于深度拷贝最佳模型
from evalutate_lstm_lko import evaluate_lstm_lko
from generate_lstm_data import generate_lstm_data
from lstm_loss_function import lstm_loss_function
from lko_lstm_network import LKO_LSTM_Network


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
    hidden_size = params['hidden_size']
    output_size = params['output_size']
    initialLearnRate = params['initialLearnRate']
    minLearnRate = params['minLearnRate']
    num_epochs = params['num_epochs']
    L1, L2, L3 = params['L1'], params['L2'], params['L3']
    batchSize = params['batchSize']
    patience = params['patience']
    lrReduceFactor = params['lrReduceFactor']

    # 2. 检查GPU可用性并初始化
    if torch.cuda.is_available():
        print('检测到可用GPU，启用加速')
        device = torch.device('cuda')
    else:
        print('未检测到GPU，使用CPU')
        device = torch.device('cpu')

    # 3. 训练数据准备
    # 从字典中获取 NumPy 数组
    control_train_np = train_data['control']
    state_train_np = train_data['state']
    label_train_np = train_data['label']

    # 将数据转换为PyTorch张量，并转换维度以匹配 (batch, ...) 格式
    # 注意：数据首先留在CPU上，在训练循环中按批次移动到GPU，以节省显存
    # state: (d, num_samples, time_step) -> (num_samples, time_step, d)
    state_train = torch.from_numpy(np.transpose(state_train_np, (1, 2, 0))).float().to(device)
    # control: (c, pred_step, num_samples) -> (num_samples, pred_step, c)
    control_train = torch.from_numpy(np.transpose(control_train_np, (2, 1, 0))).float().to(device)
    # label: (d, pred_step, num_samples, time_step) -> (num_samples, pred_step, time_step, d)
    label_train = torch.from_numpy(np.transpose(label_train_np, (2, 1, 3, 0))).float().to(device)

    num_samples = state_train.shape[0]

    # 4. 网络初始化
    net = LKO_LSTM_Network(state_size, hidden_size, output_size, control_size, delay_step)
    net.to(device)

    # 5. 训练设置
    optimizer = optim.Adam(net.parameters(), lr=initialLearnRate,  weight_decay=1e-2)
    best_test_loss = float('inf')
    best_train_loss = float('inf')
    wait_counter = 0
    best_net_state_dict = None  # 用于存储最佳模型的权重

    # 6. 自定义训练循环
    for epoch in range(num_epochs):
        net.train()  # 设置为训练模式

        # 手动打乱数据索引，模拟 MATLAB 的 shuffle(ds_train)
        shuffled_indices = np.random.permutation(num_samples)

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
            total_loss = lstm_loss_function(net, state_batch, control_batch, label_batch, L1, L2, L3)
            total_loss.backward()

            # 更新参数 (Adam优化器)
            optimizer.step()

        # 7. 评估
        net.eval()  # 设置为评估模式
        test_loss_list = []

        # MATLAB代码中只测试了第6个，这里我们遍历所有测试集
        # test_data 是一个字典列表
        for test_set in test_data:
            control_test_raw = test_set['control']
            state_test_raw = test_set['state']
            # label_test_raw = test_set['label'] # label在评估时从state_test_raw中导出

            # 为评估准备数据
            pred_step_eval = control_test_raw.shape[1] - delay_step  # 预测步数由测试控制信号的长度决定
            control_test_raw = control_test_raw[:, 0:pred_step_eval]
            initial_state_eval = torch.from_numpy(state_test_raw[:, :delay_step]).unsqueeze(0)
            initial_state_eval = initial_state_eval.permute(0, 2, 1).float()  # (1, time_step, d)

            control_eval = torch.from_numpy(control_test_raw).permute(1, 0).float()  # (pred_step, c)

            true_labels_eval = np.zeros([state_size, pred_step_eval])
            for i in range(pred_step_eval):
                true_labels_eval[:, i] = state_test_raw[:, i+delay_step]

            true_labels_eval = torch.from_numpy(true_labels_eval).to(device)

            # 调用评估函数
            test_loss, _, _ = evaluate_lstm_lko(net, control_eval, initial_state_eval, true_labels_eval, delay_step)
            test_loss_list.append(test_loss)

        mean_test_loss = np.mean(test_loss_list)

        if mean_test_loss < best_test_loss:
            best_test_loss = mean_test_loss
            # 深度拷贝当前模型的权重，而不是复制引用
            best_net_state_dict = copy.deepcopy(net.state_dict())
            # 如果需要，可以在此处保存到文件: torch.save(net.state_dict(), 'best_net.pth')

        # 8. 学习率调度 (手动实现)
        if total_loss.item() < best_train_loss:
            best_train_loss = total_loss.item()
            wait_counter = 0
        else:
            wait_counter += 1
            if wait_counter >= patience:
                current_lr = optimizer.param_groups[0]['lr']
                new_lr = max(current_lr * lrReduceFactor, minLearnRate)
                if new_lr < current_lr:
                    print(f"训练损失无改善，学习率从 {current_lr:.6f} 降至 {new_lr:.6f}")
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = new_lr
                wait_counter = 0

        current_lr_for_print = optimizer.param_groups[0]['lr']
        print(f'Epoch {epoch + 1}/{num_epochs} | 训练集当前损失: {total_loss.item():.4f} | '
              f'测试集均方根误差: {mean_test_loss:.4f} | 学习率: {current_lr_for_print:.6f}')

    print('\n训练完成！')

    # 9. 加载最佳权重到新模型并返回
    if best_net_state_dict:
        print(f"返回在测试集上表现最佳的模型 (RMSE: {best_test_loss:.4f})")
        best_net = LKO_LSTM_Network(state_size, hidden_size, output_size, control_size, delay_step)
        best_net.load_state_dict(best_net_state_dict)
        return best_net
    else:
        print("训练过程中未产生更佳模型，返回最终模型。")
        return net
