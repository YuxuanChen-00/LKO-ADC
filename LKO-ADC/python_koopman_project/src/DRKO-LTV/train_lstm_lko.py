# 文件名: train_lstm_lko.py

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
import numpy as np
import copy
import matplotlib.pyplot as plt
import random  # <--- ADDED

# 假设这些模块已正确定义和可用
from evaluate_lstm_lko import evaluate_lstm_lko, evaluate_lstm_lko2
from LKO_lstm_Network_TimeVarying import LKO_lstm_Network_TimeVarying
from lstm_loss_function import lstm_loss_function, lstm_loss_function2
from lstm_dataloader import CustomTimeSeriesDataset


# <--- ADDED: 为 DataLoader 的 worker 设置种子的函数 ---
def seed_worker(worker_id):
    """
    为 DataLoader 的 worker 设置种子, 确保多进程数据加载的可复现性。
    """
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


# ==============================================================================
def train_lstm_lko(params, train_data, test_data):
    """
    使用 DataLoader 优化的 train_lstm_lko 函数。
    """
    # 1. 参数设置 (与原版相同, 但增加了 seed 的提取)
    # ... (state_size, delay_step 等参数提取与原版相同) ...
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
    seed = params['seed']  # <--- ADDED: 从参数字典中获取当前种子

    # 3. 数据准备 (使用 DataLoader)
    train_dataset = CustomTimeSeriesDataset(train_data)

    # <--- ADDED: 为 DataLoader 创建一个确定性的随机数生成器 ---
    g = torch.Generator()
    g.manual_seed(seed)

    # <--- MODIFIED: DataLoader 的创建，增加了 worker_init_fn 和 generator ---
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batchSize,
        shuffle=True,
        num_workers=16,  # 在Windows或Jupyter中不稳定时可设为0
        pin_memory=True,
        drop_last=True,
        worker_init_fn=seed_worker,  # <--- ADDED
        generator=g,  # <--- ADDED
    )

    # 4. 网络初始化 (与原版相同)
    # 注意: 由于主函数中已调用 set_seed, 此处的权重初始化已经是可复现的
    net = LKO_lstm_Network_TimeVarying(state_size, hidden_size_lstm, hidden_size_mlp, output_size, control_size, delay_step)
    net.to(device)

    # 5. 训练设置 (与原版相同)
    optimizer = optim.Adam(net.parameters(), lr=initialLearnRate, weight_decay=1e-5)
    scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=minLearnRate)

    best_test_loss = float('inf')
    wait_counter = 0
    best_net_state_dict = None

    # 6. 自定义训练循环 (与原版相同)
    # print("开始使用 DataLoader 进行训练...") # 此打印信息可以移至主循环中
    for epoch in range(num_epochs):
        net.train()
        epoch_train_loss = 0.0

        for state_batch, control_batch, label_batch in train_loader:
            state_batch = state_batch.to(device)
            control_batch = control_batch.to(device)
            label_batch = label_batch.to(device)

            optimizer.zero_grad()
            total_loss = lstm_loss_function2(net, state_batch, control_batch, label_batch, L1, L2, L3)
            total_loss.backward()
            optimizer.step()

            epoch_train_loss += total_loss.item()

        scheduler.step()
        avg_train_loss = epoch_train_loss / len(train_loader)

        # 7. 评估 (与原版相同)
        if (epoch + 1) % 5 == 0:  # 评估频率可以降低，以加速训练
            net.eval()
            test_loss_list = []
            for test_set in test_data:
                control_test = test_set['control']
                state_test = test_set['state']
                label_test = test_set['label']
                initial_state_sequence = state_test[10 - delay_step, :, :]

                with torch.no_grad():
                    test_loss, _, _ = evaluate_lstm_lko2(net, control_test[10 - delay_step:],
                                                        initial_state_sequence,
                                                        label_test[10 - delay_step:], params_state, is_norm)
                test_loss_list.append(test_loss)

            mean_test_loss = np.mean(test_loss_list) if test_loss_list else float('inf')

            if mean_test_loss < best_test_loss:
                best_test_loss = mean_test_loss
                best_net_state_dict = copy.deepcopy(net.state_dict())
                wait_counter = 0
            else:
                wait_counter += 1

            norm_A = torch.linalg.norm(net.A)
            norm_B = torch.linalg.norm(net.B)
            # 在长训练中，减少打印频率可以使日志更清晰
            print(f'Epoch {epoch + 1}/{num_epochs} | 训练损失: {avg_train_loss:.4f} | '
                  f'测试RMSE: {mean_test_loss:.4f} | LR: {scheduler.get_last_lr()[0]:.6f} | A: {norm_A:.4f} | B: {norm_B:.4f}')

    print(f'\n训练完成! 最佳测试RMSE为: {best_test_loss:.4f}')

    # 9. 加载最佳模型并返回 (与原版相同)
    if best_net_state_dict:
        best_net = LKO_lstm_Network_TimeVarying(state_size, hidden_size_lstm, hidden_size_mlp, output_size, control_size,
                                    delay_step)
        best_net.load_state_dict(best_net_state_dict)
        return best_net
    else:
        return net
