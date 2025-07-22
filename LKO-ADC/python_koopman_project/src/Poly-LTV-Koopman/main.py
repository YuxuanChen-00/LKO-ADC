import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import copy
import random
import time
from pathlib import Path
from scipy.io import loadmat
import matplotlib

matplotlib.use('Agg')  # 使用非交互式后端，防止在服务器上出错
import matplotlib.pyplot as plt

# --- 导入您项目中的模块 ---
from model_poly_ltv import ResidualKoopmanNetwork, residual_koopman_loss
from evaluate import evaluate_with_pre_lifted_state
from dataloader import generate_koopman_data, CustomTimeSeriesDataset
from poly_lift import polynomial_expansion_td
from model_poly_lti import predict_multistep_koopman, calculate_koopman_operator


def set_global_seed(seed):
    """设置全局随机种子以确保实验的可复现性。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def seed_worker(worker_id):
    """为 DataLoader 的 worker 设置种子。"""
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


# ==================== 主训练函数 ====================
def train_residual_ltv_model(params, train_files, test_files):
    """为 ResidualKoopmanNetwork 设计的完整训练与评估流程。"""
    # 0. 设置种子和设备
    set_global_seed(params['seed'])
    device = torch.device(params['device'])
    p = params  # 使用简写

    # 1. 数据加载 和 LTI算子预计算
    print("## 1. 加载训练数据并计算静态LTI算子... ##")
    control_data_list, state_data_list, label_data_list = [], [], []
    for file_path in train_files:
        data = loadmat(file_path)
        control_win, state_win, label_win = generate_koopman_data(
            data[p['control_var_name']], data[p['state_var_name']], p['delay_time'], pred_step=1
        )
        control_data_list.append(control_win)
        state_data_list.append(state_win)
        label_data_list.append(label_win)

    control_win_all = np.concatenate(control_data_list, axis=0)
    state_win_all = np.concatenate(state_data_list, axis=0)
    label_win_all = np.concatenate(label_data_list, axis=0).squeeze(axis=1)

    num_samples = state_win_all.shape[0]
    state_all_2d = state_win_all.reshape(num_samples, -1).T
    label_all_2d = label_win_all.reshape(num_samples, -1).T
    control_all_2d = control_win_all[:, 0, -1, :].T

    state_lifted = polynomial_expansion_td(state_all_2d, p['target_dim'], p['delay_time'])
    label_lifted = polynomial_expansion_td(label_all_2d, p['target_dim'], p['delay_time'])
    A_static_np, B_static_np = calculate_koopman_operator(control_all_2d, state_lifted, label_lifted)
    print("静态LTI算子 A_static, B_static 计算完成。")

    evaluate_lti_baseline(A_static_np, B_static_np, test_files, p)

    train_data_dict = {'state': state_win_all, 'control': control_win_all, 'label': label_win_all}
    train_dataset = CustomTimeSeriesDataset(train_data_dict)
    g = torch.Generator()
    g.manual_seed(p['seed'])
    train_loader = DataLoader(
        dataset=train_dataset, batch_size=p['batch_size'], shuffle=True,
        num_workers=4, pin_memory=True, drop_last=True, worker_init_fn=seed_worker, generator=g
    )
    print(f"\n## 2. Pytorch DataLoader准备就绪，共 {len(train_dataset)} 个训练样本。##")

    # 3. 模型初始化
    model_save_path = Path(p['model_save_path'])
    best_model_filename = model_save_path / "best_residual_koopman_model.pth"
    A_static = torch.from_numpy(A_static_np).float()
    B_static = torch.from_numpy(B_static_np).float()
    C_static = torch.zeros(p['state_dim'], p['g_dim'])
    for i in range(p['state_dim']): C_static[i, i] = 1.0

    model = ResidualKoopmanNetwork(
        A_static, B_static, C_static, p['state_dim'], p['control_dim'],
        p['delta_rnn_hidden'], p['delta_mlp_hidden']
    ).to(device)
    print("## 3. ResidualKoopmanNetwork 模型初始化完成。##")

    # 4. 训练设置
    optimizer = optim.Adam(model.parameters(), lr=p['lr'])
    scheduler = CosineAnnealingLR(optimizer, T_max=p['num_epochs'], eta_min=1e-6)
    best_test_rmse = float('inf')
    best_model_state = None

    # 5. 训练循环
    print("\n## 4. 开始训练 LTV 残差部分... ##")
    start_time = time.time()
    for epoch in range(p['num_epochs']):
        model.train()
        epoch_train_loss = 0.0
        for state_history, control_win, label_state_one_step in train_loader:
            state_history = state_history.to(device)
            control_win = control_win.to(device)
            label_state_one_step = label_state_one_step.to(device)

            control_current = control_win[:, 0, -1, :]
            control_history = control_win[:, 0, :, :]
            batch_size_curr = state_history.shape[0]
            state_history_np = state_history.cpu().numpy().transpose(0, 2, 1).reshape(batch_size_curr, -1).T
            phi_current_np = polynomial_expansion_td(state_history_np, p['target_dim'], p['delay_time'])
            phi_current = torch.from_numpy(phi_current_np.T).float().to(device)

            optimizer.zero_grad()
            total_loss = residual_koopman_loss(model, phi_current, control_current, state_history, control_history,
                                               label_state_one_step, p['L1_weight'], p['L_delta_weight'])
            total_loss.backward()
            optimizer.step()
            epoch_train_loss += total_loss.item()

        scheduler.step()
        avg_train_loss = epoch_train_loss / len(train_loader)

        # 6. 周期性评估
        if (epoch + 1) % p['eval_interval'] == 0:
            model.eval()
            test_rmse_list = []
            for test_file_path in test_files:
                test_data_mat = loadmat(test_file_path)
                full_test_control = torch.from_numpy(test_data_mat[p['control_var_name']]).float().T
                full_test_state = torch.from_numpy(test_data_mat[p['state_var_name']]).float().T

                start_eval = p['predict_start_offset']
                horizon_eval = p['predict_horizon']

                past_s = full_test_state[start_eval - p['delay_time']: start_eval]
                past_c = full_test_control[start_eval - p['delay_time']: start_eval]
                future_c = full_test_control[start_eval: start_eval + horizon_eval]
                future_l = full_test_state[start_eval: start_eval + horizon_eval]

                past_s_np = past_s.cpu().numpy().T.reshape(-1, 1)
                lifted_state_np = polynomial_expansion_td(past_s_np, p['target_dim'], p['delay_time']).flatten()
                initial_lifted = torch.from_numpy(lifted_state_np).float()

                eval_params = {'is_norm': False}

                with torch.no_grad():
                    rmse, _, _ = evaluate_with_pre_lifted_state(model, initial_lifted, past_s, past_c, future_c,
                                                                future_l, eval_params)
                test_rmse_list.append(rmse)

            mean_test_rmse = np.mean(test_rmse_list)
            print(
                f'Epoch {epoch + 1}/{p["num_epochs"]} | 训练损失: {avg_train_loss:.6f} | 测试RMSE: {mean_test_rmse:.6f} | LR: {scheduler.get_last_lr()[0]:.6f}')

            if mean_test_rmse < best_test_rmse:
                best_test_rmse = mean_test_rmse
                best_model_state = copy.deepcopy(model.state_dict())
                torch.save(best_model_state, best_model_filename)
                print(f'  -> 新的最佳RMSE! 模型已保存至 {best_model_filename}')

    end_time = time.time()
    print(f'\n## 5. 训练完成! ##')
    print(f'总耗时: {(end_time - start_time) / 60:.2f} 分钟')
    print(f'最佳测试集平均RMSE为: {best_test_rmse:.6f}')

    return best_model_filename


def main():
    """主函数， orchestrates 整个训练和评估流程。"""
    # ==========================================================================
    # 1. 参数设置
    # ==========================================================================
    current_dir = Path(__file__).resolve().parent
    base_data_path = current_dir.parent.parent / "data" / "SorotokiData" / "MotionData8" / "FilteredDataPos"
    model_save_path = current_dir / "models" / "Poly_LTV_Koopman"
    model_save_path.mkdir(parents=True, exist_ok=True)
    print(f"模型将保存在: '{model_save_path.resolve()}'")

    params = {
        'seed': 42,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'control_var_name': 'input',
        'state_var_name': 'state',
        'train_data_path': str(base_data_path / "80minTrain"),
        'test_data_path': str(base_data_path / "50secTest"),
        'model_save_path': str(model_save_path),
        'state_dim': 9,
        'control_dim': 3,
        'delay_time': 7,
        'target_dim': 30,
        'delta_rnn_hidden': 64,
        'delta_mlp_hidden': 128,
        'lr': 1e-3,
        'num_epochs': 200,
        'batch_size': 128,
        'eval_interval': 5,
        'L1_weight': 1.0,
        'L_delta_weight': 0.01,
        'predict_start_offset': 50,
        'predict_horizon': 100,
    }
    params['g_dim'] = params['target_dim'] * params['delay_time']
    params['state_eval_slice'] = slice(0, params['state_dim'])

    # ==========================================================================
    # 2. 启动训练
    # ==========================================================================
    train_path = Path(params['train_data_path'])
    test_path = Path(params['test_data_path'])
    train_files = sorted(list(train_path.glob('*.mat')))
    test_files = sorted(list(test_path.glob('*.mat')))

    """为 ResidualKoopmanNetwork 设计的完整训练与评估流程。"""
    # 0. 设置种子和设备
    set_global_seed(params['seed'])
    device = torch.device(params['device'])

    # 1. 数据加载 和 LTI算子预计算
    print("## 1. 加载训练数据并计算静态LTI算子... ##")
    control_data_list, state_data_list, label_data_list = [], [], []
    for file_path in train_files:
        data = loadmat(file_path)
        control_win, state_win, label_win = generate_koopman_data(
            data[p['control_var_name']], data[p['state_var_name']], p['delay_time'], pred_step=1
        )
        control_data_list.append(control_win)
        state_data_list.append(state_win)
        label_data_list.append(label_win)

    control_win_all = np.concatenate(control_data_list, axis=0)
    state_win_all = np.concatenate(state_data_list, axis=0)
    label_win_all = np.concatenate(label_data_list, axis=0).squeeze(axis=1)

    num_samples = state_win_all.shape[0]
    state_all_2d = state_win_all.reshape(num_samples, -1).T
    label_all_2d = label_win_all.reshape(num_samples, -1).T
    control_all_2d = control_win_all[:, 0, -1, :].T

    state_lifted = polynomial_expansion_td(state_all_2d, params['target_dim'], params['delay_time'])
    label_lifted = polynomial_expansion_td(label_all_2d, params['target_dim'], params['delay_time'])
    A_static_np, B_static_np = calculate_koopman_operator(control_all_2d, state_lifted, label_lifted)
    print("静态LTI算子 A_static, B_static 计算完成。")














    best_model_path = train_residual_ltv_model(params, train_files, test_files)

    # ==========================================================================
    # 3. 对最佳模型进行最终评估和绘图
    # ==========================================================================
    print(f"\n## 6. 对最佳模型 '{best_model_path}' 进行最终评估与绘图... ##")
    # 重新加载最佳模型
    A_static_dummy = torch.zeros(params['g_dim'], params['g_dim'])
    B_static_dummy = torch.zeros(params['g_dim'], params['control_dim'])
    C_static_dummy = torch.zeros(params['state_dim'], params['g_dim'])
    final_model = ResidualKoopmanNetwork(
        A_static_dummy, B_static_dummy, C_static_dummy, params['state_dim'], params['control_dim'],
        params['delta_rnn_hidden'], params['delta_mlp_hidden']
    )
    final_model.load_state_dict(torch.load(best_model_path, map_location=params['device']))
    final_model.to(params['device'])
    final_model.eval()

    final_rmse_scores = []
    for i, test_file_path in enumerate(test_files):
        test_data_mat = loadmat(test_file_path)
        full_test_control = torch.from_numpy(test_data_mat[params['control_var_name']]).float().T
        full_test_state = torch.from_numpy(test_data_mat[params['state_var_name']]).float().T

        start = params['predict_start_offset']
        horizon = params['predict_horizon']

        past_s = full_test_state[start - params['delay_time']: start]
        past_c = full_test_control[start - params['delay_time']: start]
        future_c = full_test_control[start: start + horizon]
        future_l = full_test_state[start: start + horizon]

        past_s_np = past_s.cpu().numpy().T.reshape(-1, 1)
        lifted_state_np = polynomial_expansion_td(past_s_np, params['target_dim'], params['delay_time']).flatten()
        initial_lifted = torch.from_numpy(lifted_state_np).float()

        eval_params = {'is_norm': False}
        with torch.no_grad():
            rmse_score, y_true, y_pred = evaluate_with_pre_lifted_state(final_model, initial_lifted, past_s, past_c,
                                                                        future_c, future_l, eval_params)

        final_rmse_scores.append(rmse_score)

        fig = plt.figure(figsize=(18, 10))
        fig.suptitle(f'Poly-LTV模型 - 测试轨迹 {i + 1} - 真实值 vs. 预测值 (RMSE: {rmse_score:.4f})', fontsize=16)
        time_axis = np.arange(y_true.shape[1])
        for j in range(params['state_dim']):
            ax = plt.subplot(3, 3, j + 1)
            ax.plot(time_axis, y_true[j, :], 'b-', linewidth=1.5, label='True')
            ax.plot(time_axis, y_pred[j, :], 'r--', linewidth=1.5, label='Predicted')
            ax.set_title(f'状态维度 {j + 1}')
            ax.set_xlabel('Time Step')
            ax.grid(True, linestyle='--', alpha=0.6)
            if j == 0: ax.legend()
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plot_filename = model_save_path / f'final_evaluation_trajectory_{i + 1}.png'
        plt.savefig(plot_filename)
        plt.close(fig)
        print(f"评估图已保存: {plot_filename}")

    mean_final_rmse = np.mean(final_rmse_scores)
    print(f'\n--- Poly-LTV模型在所有测试轨迹上的最终平均RMSE: {mean_final_rmse:.4f} ---')


if __name__ == '__main__':
    main()