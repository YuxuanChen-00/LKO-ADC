import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR
import numpy as np
import copy  # 用于深度拷贝最佳模型
import matplotlib.pyplot as plt  # 导入 matplotlib 库

# 假设这些模块已正确定义和可用
from evalutate_mlp_lko import evaluate_mlp_lko
from generate_mlp_data import generate_mlp_data
from mlp_loss_function import mlp_loss_function
from lko_mlp_network import LKO_mlp_Network


# ==============================================================================
def train_mlp_lko(params, train_data, test_data):
    """
    MATLAB函数 train_mlp_lko 的直接Python/PyTorch转换。

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
    net = LKO_mlp_Network(state_size, hidden_size, output_size, control_size, delay_step)
    net.to(device)

    # 5. 训练设置
    optimizer = optim.Adam(net.parameters(), lr=initialLearnRate, weight_decay=1e-4)
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
            total_loss = mlp_loss_function(net, state_batch, control_batch, label_batch, L1, L2, L3)
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

        # MATLAB代码中只测试了第6个，这里我们遍历所有测试集
        # test_data 是一个字典列表
        for test_set in test_data:
            control_test_raw = test_set['control']
            state_test_raw = test_set['state']
            # label_test_raw = test_set['label'] # label在评估时从state_test_raw中导出

            # 为评估准备数据
            # 确保 pred_step_eval 不会是负数
            pred_step_eval = control_test_raw.shape[1] - delay_step
            if pred_step_eval <= 0:
                print(
                    f"警告：测试集控制信号长度 ({control_test_raw.shape[1]}) 小于或等于延迟步长 ({delay_step})。跳过此测试集。")
                continue  # 跳过当前测试集，避免错误

            control_test_raw = control_test_raw[:, 0:pred_step_eval]
            initial_state_eval = torch.from_numpy(state_test_raw[:, :delay_step]).unsqueeze(0)
            initial_state_eval = initial_state_eval.permute(0, 2, 1).float()  # (1, time_step, d)

            control_eval = torch.from_numpy(control_test_raw).permute(1, 0).float()  # (pred_step, c)

            true_labels_eval = np.zeros([state_size, pred_step_eval])
            for i in range(pred_step_eval):
                true_labels_eval[:, i] = state_test_raw[:, i + delay_step]

            true_labels_eval = torch.from_numpy(true_labels_eval).to(device)

            # 调用评估函数
            with torch.no_grad():  # 评估时禁用梯度计算
                test_loss, _, _ = evaluate_mlp_lko(net, control_eval.to(device), initial_state_eval.to(device),
                                                    true_labels_eval.to(device), delay_step)
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

        # 8. 学习率调度 (手动实现)
        # 你的手动学习率衰减逻辑被 CosineAnnealingLR 替代，但早停逻辑可以保留
        # 如果你希望在余弦退火的同时有基于性能的早停，这里是逻辑：
        if wait_counter >= patience:
            print(f"测试损失在 {patience} 个 epoch 内没有改善，提前停止训练。")
            break  # 提前结束训练循环

        current_lr_for_print = optimizer.param_groups[0]['lr']

        print(f'Epoch {epoch + 1}/{num_epochs} | 训练集当前损失: {avg_train_loss:.4f} | '
              f'测试集均方根误差: {mean_test_loss:.4f} | 学习率: {current_lr_for_print:.6f}')

    print('\n训练完成！')

    # 9. 绘制损失曲线
    plt.figure(figsize=(12, 6))
    plt.plot(train_losses, label='训练集损失')
    plt.plot(test_losses, label='测试集RMSE')
    plt.xlabel('Epoch')
    plt.ylabel('损失值')
    plt.title('训练和测试损失曲线')
    plt.legend()
    plt.grid(True)
    plt.show()

    # 绘制学习率曲线
    plt.figure(figsize=(12, 6))
    plt.plot(learning_rates, label='学习率')
    plt.xlabel('Epoch')
    plt.ylabel('学习率')
    plt.title('学习率变化曲线')
    plt.legend()
    plt.grid(True)
    plt.show()

    # 10. 加载最佳权重到新模型并返回
    if best_net_state_dict:
        print(f"返回在测试集上表现最佳的模型 (RMSE: {best_test_loss:.4f})")
        best_net = LKO_mlp_Network(state_size, hidden_size, output_size, control_size, delay_step)
        best_net.load_state_dict(best_net_state_dict)
        return best_net
    else:
        print("训练过程中未产生更佳模型，返回最终模型。")
        return net
