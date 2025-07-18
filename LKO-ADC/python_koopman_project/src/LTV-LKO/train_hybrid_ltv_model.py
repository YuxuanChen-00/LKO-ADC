import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
import numpy as np
import copy
import random
from pathlib import Path

# --- 导入您已经准备好的、最新的模块和函数 ---
from model_hybrid_ltv import HybridLTVKoopmanNetwork
from loss_function_hybrid import hybrid_ltv_loss
from evaluate_hybrid_model import evaluate_hybrid_ltv_model
from dataloader import CustomTimeSeriesDataset


# ==================== 新增: 全局种子设置函数 ====================
def set_global_seed(seed):
    """
    设置全局随机种子以确保实验的可复现性。
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # 以下两行是为了确保在使用cuDNN时结果也是确定的
    # 注意：这可能会对性能产生微小的影响
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ===============================================================


def seed_worker(worker_id):
    """为 DataLoader 的 worker 设置种子。"""
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def train_hybrid_ltv_model(params, train_data, test_data):
    """
    为 HybridLTVKoopmanNetwork 设计的、支持精细分阶段训练的函数。
    """
    # 0. 设置全局种子
    seed = params['seed']
    set_global_seed(seed)

    # 1. 参数设置
    # --- 提取模型架构参数 ---
    state_size = params['state_size']
    control_size = params['control_size']
    delay_step = params['delay_step']
    g_dim = params['PhiDimensions']
    encoder_gru_hidden = params['encoder_gru_hidden']
    encoder_mlp_hidden = params['encoder_mlp_hidden']
    delta_rnn_hidden = params['delta_rnn_hidden']
    delta_mlp_hidden = params['delta_mlp_hidden']

    # --- 提取训练超参数 ---
    initialLearnRate = params['initialLearnRate']
    minLearnRate = params['minLearnRate']
    num_epochs = params['num_epochs']
    L1, L2, L3, L_delta = params['L1'], params['L2'], params['L3'], params['L_delta']
    batchSize = params['batchSize']
    patience = params['patience']
    device = params['device']
    train_mode = params['train_mode']
    pretrained_path = params['pretrained_path']

    # --- 提取评估所需参数 ---
    params_state = params['params_state']
    is_norm = params['is_norm']

    # 2. 数据加载器
    train_dataset = CustomTimeSeriesDataset(train_data)
    g = torch.Generator()
    g.manual_seed(seed)
    train_loader = DataLoader(
        dataset=train_dataset, batch_size=batchSize, shuffle=True,
        num_workers=8, pin_memory=True, drop_last=True,
        worker_init_fn=seed_worker, generator=g,
    )

    # 3. 网络初始化
    net = HybridLTVKoopmanNetwork(
        state_size=state_size, control_size=control_size, time_step=delay_step, g_dim=g_dim,
        encoder_gru_hidden=encoder_gru_hidden, encoder_mlp_hidden=encoder_mlp_hidden,
        delta_rnn_hidden=delta_rnn_hidden, delta_mlp_hidden=delta_mlp_hidden
    )

    # 精细的参数冻结逻辑
    print(f"--- 当前训练模式: {train_mode} ---")
    for param in net.parameters():
        param.requires_grad = True

    if train_mode == 'stage1':
        for param in net.delta_encoder.parameters():
            param.requires_grad = False
        for param in net.delta_generator.parameters():
            param.requires_grad = False
        with torch.no_grad():
            for param in net.delta_encoder.parameters():
                param.zero_()
            for param in net.delta_generator.parameters():
                param.zero_()
        print("模式[stage1]: 已冻结时变部分(delta*)，仅训练LTI基线模型。")
        L_delta = 0.0

    elif train_mode == 'stage2':
        pretrained_path = params.get(pretrained_path)
        if pretrained_path and Path(pretrained_path).exists():
            print(f"正在从 '{pretrained_path}' 加载预训练模型...")
            net.load_state_dict(torch.load(pretrained_path, map_location=device))

        for param in net.base_gru.parameters():
            param.requires_grad = False
        for param in net.base_mlp.parameters():
            param.requires_grad = False
        for param in net.A_static_layer.parameters():
            param.requires_grad = False
        for param in net.B_static_layer.parameters():
            param.requires_grad = False
        for param in net.C.parameters():
            param.requires_grad = False
        print("模式[stage2]: 已冻结LTI基线模型和编码器，仅训练时变(delta*)部分。")
    else:
        print("模式[end-to-end]: 训练网络中的所有参数。")

    # 4. 训练设置
    trainable_params = filter(lambda p: p.requires_grad, net.parameters())
    optimizer = optim.Adam(trainable_params, lr=initialLearnRate, weight_decay=1e-5)
    scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=minLearnRate)

    best_test_loss = float('inf')
    wait_counter = 0
    best_net_state_dict = None
    net.to(device)

    # 5. 训练循环
    for epoch in range(num_epochs):
        net.train()
        epoch_train_loss = 0.0
        for state_batch, control_batch, label_batch in train_loader:
            state_batch, control_batch, label_batch = state_batch.to(device), control_batch.to(device), label_batch.to(
                device)
            optimizer.zero_grad()
            total_loss = hybrid_ltv_loss(net, state_batch, control_batch, label_batch, L1, L2, L3, L_delta)
            total_loss.backward()
            optimizer.step()
            epoch_train_loss += total_loss.item()
        scheduler.step()
        avg_train_loss = epoch_train_loss / len(train_loader)

        # 6. 评估
        if (epoch + 1) % 1 == 0:
            net.eval()
            test_loss_list = []
            for test_set in test_data:
                control_test, state_test, label_test = test_set['control'], test_set['state'], test_set['label']
                start_index = 10 - delay_step
                initial_state_sequence = state_test[start_index]
                initial_control_sequence = control_test[start_index, 0]
                future_labels_for_eval = label_test[start_index:, 0]
                future_controls_for_eval = control_test[start_index:, 0]
                with torch.no_grad():
                    test_loss, _, _ = evaluate_hybrid_ltv_model(
                        net, initial_state_sequence, initial_control_sequence,
                        future_labels_for_eval, future_controls_for_eval,
                        params_state, is_norm
                    )
                test_loss_list.append(test_loss)
            mean_test_loss = np.mean(test_loss_list) if test_loss_list else float('inf')

            if mean_test_loss < best_test_loss:
                best_test_loss = mean_test_loss
                best_net_state_dict = copy.deepcopy(net.state_dict())
                wait_counter = 0
            else:
                wait_counter += 1

            print(f'Epoch {epoch + 1}/{num_epochs} | 模式: {train_mode} | 训练损失: {avg_train_loss:.4f} | '
                  f'测试RMSE: {mean_test_loss:.4f} | LR: {scheduler.get_last_lr()[0]:.6f}')

    print(f'\n训练完成! 模式 {train_mode} 的最佳测试RMSE为: {best_test_loss:.4f}')

    # 7. 返回最佳模型
    if best_net_state_dict:
        net.load_state_dict(best_net_state_dict)
    return net


def train_two_stage_hybrid_model(params, train_data, test_data):
    """
    一个完整的、自动执行两阶段训练的函数。
    增加了在训练日志中实时输出各算子范数的功能。
    """
    # 0. 设置全局种子
    seed = params['seed']
    set_global_seed(seed)

    # 1. 参数设置
    state_size = params['state_size']
    control_size = params['control_size']
    delay_step = params['delay_step']
    g_dim = params['PhiDimensions']
    encoder_gru_hidden = params['encoder_gru_hidden']
    encoder_mlp_hidden = params['encoder_mlp_hidden']
    delta_rnn_hidden = params['delta_rnn_hidden']
    delta_mlp_hidden = params['delta_mlp_hidden']
    minLearnRate = params['minLearnRate']
    L1, L2, L3 = params['L1'], params['L2'], params['L3']
    batchSize = params['batchSize']
    device = params['device']
    params_state = params['params_state']
    is_norm = params['is_norm']
    num_epochs_s1, lr_s1 = params['num_epochs_s1'], params['lr_s1']
    num_epochs_s2, lr_s2 = params['num_epochs_s2'], params['lr_s2']
    L_delta_s2 = params['L_delta']

    # 2. 数据加载器
    train_dataset = CustomTimeSeriesDataset(train_data)
    g = torch.Generator()
    g.manual_seed(seed)
    train_loader = DataLoader(
        dataset=train_dataset, batch_size=batchSize, shuffle=True,
        num_workers=8, pin_memory=True, drop_last=True,
        worker_init_fn=seed_worker, generator=g,
    )

    # 3. 网络初始化 (现在是可复现的)
    net = HybridLTVKoopmanNetwork(
        state_size=state_size, control_size=control_size, time_step=delay_step, g_dim=g_dim,
        encoder_gru_hidden=encoder_gru_hidden, encoder_mlp_hidden=encoder_mlp_hidden,
        delta_rnn_hidden=delta_rnn_hidden, delta_mlp_hidden=delta_mlp_hidden
    )
    net.to(device)

    # ==============================================================================
    #                             第一阶段: 训练LTI基线模型
    # ==============================================================================
    print("\n" + "=" * 80)
    print("                      开始第一阶段: 训练LTI基线模型")
    print("=" * 80)

    for param in net.parameters():
        param.requires_grad = True
    for param in net.delta_encoder.parameters():
        param.requires_grad = False
    for param in net.delta_generator.parameters():
        param.requires_grad = False
    with torch.no_grad():
        for param in net.delta_encoder.parameters():
            param.zero_()
        for param in net.delta_generator.parameters():
            param.zero_()
    print("模式[stage1]: 已冻结并初始化时变部分(delta*)，仅训练LTI基线模型。")

    trainable_params_s1 = filter(lambda p: p.requires_grad, net.parameters())
    optimizer_s1 = optim.Adam(trainable_params_s1, lr=lr_s1, weight_decay=1e-5)
    scheduler_s1 = CosineAnnealingLR(optimizer_s1, T_max=num_epochs_s1, eta_min=minLearnRate)

    best_lti_state_dict = None
    best_lti_loss = float('inf')

    for epoch in range(num_epochs_s1):
        net.train()
        epoch_train_loss = 0.0
        for state_batch, control_batch, label_batch in train_loader:
            state_batch, control_batch, label_batch = state_batch.to(device), control_batch.to(device), label_batch.to(
                device)
            optimizer_s1.zero_grad()
            total_loss = hybrid_ltv_loss(net, state_batch, control_batch, label_batch, L1, L2, L3, 0.0)
            total_loss.backward()
            optimizer_s1.step()
            epoch_train_loss += total_loss.item()
        scheduler_s1.step()
        avg_train_loss = epoch_train_loss / len(train_loader)

        if (epoch + 1) % 5 == 0:
            net.eval()
            test_loss_list = []
            for test_set in test_data:
                control_test, state_test, label_test = test_set['control'], test_set['state'], test_set['label']
                start_index = 10 - delay_step
                initial_state_sequence = state_test[start_index]
                initial_control_sequence = control_test[start_index, 0]
                future_labels_for_eval = label_test[start_index:, 0]
                future_controls_for_eval = control_test[start_index:, 0]
                with torch.no_grad():
                    test_loss, _, _ = evaluate_hybrid_ltv_model(net, initial_state_sequence, initial_control_sequence,
                                                                future_labels_for_eval, future_controls_for_eval,
                                                                params_state, is_norm)
                test_loss_list.append(test_loss)
            mean_test_loss = np.mean(test_loss_list) if test_loss_list else float('inf')

            if mean_test_loss < best_lti_loss:
                best_lti_loss = mean_test_loss
                best_lti_state_dict = copy.deepcopy(net.state_dict())

            with torch.no_grad():
                norm_A_s = torch.linalg.norm(net.A_static_layer.weight).item()
                norm_B_s = torch.linalg.norm(net.B_static_layer.weight).item()
                norm_C_s = torch.linalg.norm(net.C_static_layer.weight).item()
                sample_state_hist = test_data[0]['state'][10 - delay_step].unsqueeze(0).to(device)
                sample_control_hist = test_data[0]['control'][10 - delay_step, 0].unsqueeze(0).to(device)
                context = net.delta_encoder(sample_state_hist, sample_control_hist)
                d_A, d_B, d_C = net.delta_generator(context)
                norm_dA = torch.linalg.norm(d_A).item()
                norm_dB = torch.linalg.norm(d_B).item()
                norm_dC = torch.linalg.norm(d_C).item()

            print(
                f'阶段1 - Epoch {epoch + 1}/{num_epochs_s1} | '
                f'Loss: {avg_train_loss:.4f} | '
                f'RMSE: {mean_test_loss:.4f} | '
                f'N(As): {norm_A_s:.2f}, N(Bs): {norm_B_s:.2f}, N(Cs): {norm_C_s:.2f} | '
                f'N(dA): {norm_dA:.2f}, N(dB): {norm_dB:.2f}, N(dC): {norm_dC:.2f}'
            )

    print(f"\n第一阶段训练完成! 最佳LTI测试RMSE为: {best_lti_loss:.4f}")
    if not best_lti_state_dict:
        best_lti_state_dict = net.state_dict()

    # ==============================================================================
    #                           第二阶段: 训练LTV残差部分
    # ==============================================================================
    print("\n" + "=" * 80)
    print("                      开始第二阶段: 训练LTV残差模型")
    print("=" * 80)

    print("加载第一阶段训练的最佳LTI模型...")
    net.load_state_dict(best_lti_state_dict)

    for param in net.parameters():
        param.requires_grad = False
    for param in net.delta_encoder.parameters():
        param.requires_grad = True
    for param in net.delta_generator.parameters():
        param.requires_grad = True
    print("模式[stage2]: 已冻结LTI基线模型和编码器，仅训练时变(delta*)部分。")

    trainable_params_s2 = filter(lambda p: p.requires_grad, net.parameters())
    optimizer_s2 = optim.Adam(trainable_params_s2, lr=lr_s2, weight_decay=1e-5)
    scheduler_s2 = CosineAnnealingLR(optimizer_s2, T_max=num_epochs_s2, eta_min=minLearnRate)

    best_ltv_state_dict = None
    best_ltv_loss = float('inf')

    for epoch in range(num_epochs_s2):
        net.train()
        epoch_train_loss = 0.0
        for state_batch, control_batch, label_batch in train_loader:
            state_batch, control_batch, label_batch = state_batch.to(device), control_batch.to(device), label_batch.to(
                device)
            optimizer_s2.zero_grad()
            total_loss = hybrid_ltv_loss(net, state_batch, control_batch, label_batch, L1, L2, L3, L_delta_s2)
            total_loss.backward()
            optimizer_s2.step()
            epoch_train_loss += total_loss.item()
        scheduler_s2.step()
        avg_train_loss = epoch_train_loss / len(train_loader)

        if (epoch + 1) % 5 == 0:
            net.eval()
            test_loss_list = []
            for test_set in test_data:
                control_test, state_test, label_test = test_set['control'], test_set['state'], test_set['label']
                start_index = 10 - delay_step
                initial_state_sequence = state_test[start_index]
                initial_control_sequence = control_test[start_index, 0]
                future_labels_for_eval = label_test[start_index:, 0]
                future_controls_for_eval = control_test[start_index:, 0]
                with torch.no_grad():
                    test_loss, _, _ = evaluate_hybrid_ltv_model(net, initial_state_sequence, initial_control_sequence,
                                                                future_labels_for_eval, future_controls_for_eval,
                                                                params_state, is_norm)
                test_loss_list.append(test_loss)
            mean_test_loss = np.mean(test_loss_list) if test_loss_list else float('inf')

            if mean_test_loss < best_ltv_loss:
                best_ltv_loss = mean_test_loss
                best_ltv_state_dict = copy.deepcopy(net.state_dict())

            with torch.no_grad():
                norm_A_s = torch.linalg.norm(net.A_static_layer.weight).item()
                norm_B_s = torch.linalg.norm(net.B_static_layer.weight).item()
                norm_C_s = torch.linalg.norm(net.C_static_layer.weight).item()
                sample_state_hist = test_data[0]['state'][10 - delay_step].unsqueeze(0).to(device)
                sample_control_hist = test_data[0]['control'][10 - delay_step, 0].unsqueeze(0).to(device)
                context = net.delta_encoder(sample_state_hist, sample_control_hist)
                d_A, d_B, d_C = net.delta_generator(context)
                norm_dA = torch.linalg.norm(d_A).item()
                norm_dB = torch.linalg.norm(d_B).item()
                norm_dC = torch.linalg.norm(d_C).item()

            print(
                f'阶段2 - Epoch {epoch + 1}/{num_epochs_s2} | '
                f'Loss: {avg_train_loss:.4f} | '
                f'RMSE: {mean_test_loss:.4f} | '
                f'N(As): {norm_A_s:.2f}, N(Bs): {norm_B_s:.2f}, N(Cs): {norm_C_s:.2f} | '
                f'N(dA): {norm_dA:.4f}, N(dB): {norm_dB:.4f}, N(dC): {norm_dC:.4f}'
            )

    print(f"\n第二阶段训练完成! 最佳LTV测试RMSE为: {best_ltv_loss:.4f}")

    if best_ltv_state_dict:
        net.load_state_dict(best_ltv_state_dict)

    print("\n两阶段训练全部完成。返回最终模型。")
    return net, best_ltv_loss


def train_hybrid_ltv_model_e2e(params, train_data, test_data):
    """
    修改后的训练函数，用于端到端地、同时训练模型的时变与时不变部分。
    """
    # 0. 设置全局种子
    seed = params['seed']
    set_global_seed(seed)

    # 1. 参数设置
    state_size = params['state_size']
    control_size = params['control_size']
    delay_step = params['delay_step']
    g_dim = params['PhiDimensions']
    encoder_gru_hidden = params['encoder_gru_hidden']
    encoder_mlp_hidden = params['encoder_mlp_hidden']
    delta_rnn_hidden = params['delta_rnn_hidden']
    delta_mlp_hidden = params['delta_mlp_hidden']
    initialLearnRate = params['initialLearnRate']
    minLearnRate = params['minLearnRate']
    num_epochs = params['num_epochs']
    L1, L2, L3, L_delta = params['L1'], params['L2'], params['L3'], params['L_delta']
    batchSize = params['batchSize']
    device = params['device']
    params_state = params['params_state']
    is_norm = params['is_norm']

    # 2. 数据加载器
    train_dataset = CustomTimeSeriesDataset(train_data)
    g = torch.Generator