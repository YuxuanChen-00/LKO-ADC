import torch
import numpy as np
from typing import Dict, Any, Tuple
from scipy.io import loadmat

# 从您的项目文件中导入必要的模块和函数
from model_poly_ltv import ResidualKoopmanNetwork
from poly_lift import polynomial_expansion_td
from src.normalize_data import denormalize_data
from src.calculate_rmse import calculate_rmse
from model_poly_lti import predict_multistep_koopman

def evaluate_with_pre_lifted_state(
        model: ResidualKoopmanNetwork,
        initial_lifted_state: torch.Tensor,
        past_states: torch.Tensor,
        past_controls: torch.Tensor,
        future_controls: torch.Tensor,
        future_labels: torch.Tensor,
        params: Dict[str, Any]
) -> Tuple[float, np.ndarray, np.ndarray]:
    """
    接收预先计算好的升维状态和切分好的数据，使用 ResidualKoopmanNetwork 模型进行多步预测评估。
    此版本将升维步骤从函数中分离，由调用者完成。

    Args:
        model (ResidualKoopmanNetwork): 训练好的模型实例。
        initial_lifted_state (torch.Tensor): 【新】初始时刻的、已经升维后的状态, shape: (g_dim,)。
        past_states (torch.Tensor): 过去的状态历史序列, shape: (delay_time, state_dim)。
        past_controls (torch.Tensor): 过去的控制历史序列, shape: (delay_time, control_dim)。
        future_controls (torch.Tensor): 未来的控制输入, shape: (horizon, control_dim)。
        future_labels (torch.Tensor): 未来的真实状态标签, shape: (horizon, state_dim)。
        params (Dict[str, Any]): 包含评估所需参数的字典，例如:
            - 'is_norm' (bool): 是否对数据进行反归一化。
            - 'params_state' (np.ndarray): 状态数据的归一化参数 (均值和标准差)。

    Returns:
        Tuple[float, np.ndarray, np.ndarray]:
            - rmse (float): 计算出的均方根误差。
            - y_true (np.ndarray): 真实的未来状态序列 (反归一化后)。
            - y_pred (np.ndarray): 预测的未来状态序列 (反归一化后)。
    """
    # 1. 准备工作 (Setup)
    model.eval()
    device = next(model.parameters()).device

    # 2. 准备模型输入
    #    将所有输入张量移动到正确的设备，并增加批处理维度 (B=1)
    phi_initial = initial_lifted_state.to(device).unsqueeze(0)
    state_history = past_states.to(device).unsqueeze(0)
    control_history = past_controls.to(device).unsqueeze(0)
    future_control_sequence = future_controls.to(device).unsqueeze(0)

    # 【主要修改点】: 此版本不再需要内部计算 phi_initial，而是直接使用传入的参数。
    #  之前用于升维的 `polynomial_expansion_td` 调用已被移除。

    # 3. 执行多步预测
    with torch.no_grad():
        state_pred_seq, _, _ = model.predict_multistep(
            phi_initial,
            future_control_sequence,
            state_history,
            control_history
        )

    # 4. 后处理 (Post-processing)
    y_pred = state_pred_seq.squeeze(0).t()
    y_true = future_labels.t().to(device)

    #    (可选) 反归一化
    if params.get('is_norm', False):
        y_pred = denormalize_data(y_pred.T, params['params_state']).T
        y_true = denormalize_data(y_true.T, params['params_state']).T

    #    计算RMSE
    rmse = calculate_rmse(y_pred, y_true)

    return rmse, y_true.cpu().numpy(), y_pred.cpu().numpy()


def evaluate_lti_baseline(A, B, C, test_data, params):
    """使用静态的A, B算子评估LTI模型的性能。"""
    rmse_list = []
    for test_set in test_data:
        control_test = test_set['control']
        state_test = test_set['state']
        label_test = test_set['label']


        U_future = control_test[:, 0, -1, :]
        state_future = label_test[:, 0, -1, :]
        state_initial = state_test[0, -1, :]
        psi_initial = polynomial_expansion_td(state_initial, params['target_dim'], params['delay_time'])
        horizon = U_future.shape[0]

        psi_pred = predict_multistep_koopman(A, B, U_future, psi_initial, horizon)
        state_pred = C*psi_pred

        if params['is_norm']:
            state_pred = denormalize_data(state_pred, params['params_state'])
            state_future = denormalize_data(state_future, params['params_state'])

        rmse = calculate_rmse(state_pred, state_future)
        rmse_list.append(rmse)

    return rmse_list

