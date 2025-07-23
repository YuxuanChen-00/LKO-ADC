import numpy as np
from scipy.optimize import minimize
from scipy import sparse
import cvxopt
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')



def incremental_mpc(Q_cost, F_cost, R_cost, N_pred,
                    A_sys, B_sys, C_sys,
                    current_X_koopman, prev_U, Y_ref_horizon,
                    max_abs_delta_U, U_abs_min, U_abs_max,
                    n_inputs, n_outputs, n_koopman_states):
    """
    带有输入增量约束和绝对输入约束的MPC控制器 (Python实现)

    参数:
        Q_cost (np.array): 输出误差权重矩阵 (n_outputs x n_outputs)
        F_cost (np.array): 输入增量权重矩阵 (n_inputs x n_inputs)
        R_cost (np.array): 输入幅值权重矩阵 (n_inputs x n_inputs)
        N_pred (int): 预测时域长度
        A_sys (np.array): Koopman状态转移矩阵A
        B_sys (np.array): Koopman输入矩阵B
        C_sys (np.array): Koopman输出矩阵C
        current_X_koopman (np.array): 当前Koopman状态向量
        prev_U (np.array): 上一时刻控制输入 (n_inputs x 1)
        Y_ref_horizon (np.array): 预测时域内的参考输出轨迹 (n_outputs x N_pred)
        max_abs_delta_U (np.array): 输入增量最大绝对值限制 (n_inputs x 1)
        U_abs_min (np.array): 控制输入最小绝对值限制 (n_inputs x 1)
        U_abs_max (np.array): 控制输入最大绝对值限制 (n_inputs x 1)
        n_inputs (int): 控制输入数量
        n_outputs (int): 输出数量
        n_koopman_states (int): Koopman状态维度

    返回:
        np.array: 最优输入增量序列 (n_inputs x N_pred)
                  若求解失败返回零序列
    """

    # 1. 构建增广系统 [1,4](@ref)
    Aa = np.block([
        [A_sys, B_sys],
        [np.zeros((n_inputs, n_koopman_states)), np.eye(n_inputs)]
    ])
    Ba = np.vstack([B_sys, np.eye(n_inputs)])
    Ca = np.hstack([C_sys, np.zeros((n_outputs, n_inputs))])
    n_aug_states = n_koopman_states + n_inputs

    # 2. 创建预测矩阵S_z和S_delta_u [1,4](@ref)
    S_z = np.zeros((N_pred * n_aug_states, n_aug_states))
    S_delta_u = np.zeros((N_pred * n_aug_states, N_pred * n_inputs))

    temp_A_power_i = np.eye(n_aug_states)
    for i in range(N_pred):
        row_start = i * n_aug_states
        row_end = (i + 1) * n_aug_states

        temp_A_power_i = temp_A_power_i @ Aa
        S_z[row_start:row_end, :] = temp_A_power_i

        for j in range(i + 1):
            col_start = j * n_inputs
            col_end = (j + 1) * n_inputs

            power = i - j
            A_power = np.linalg.matrix_power(Aa, power) if power > 0 else np.eye(n_aug_states)
            S_delta_u[row_start:row_end, col_start:col_end] = A_power @ Ba

    # 3. 构建输出预测矩阵 [1](@ref)
    C_aug_block = np.kron(np.eye(N_pred), Ca)
    P_y = C_aug_block @ S_delta_u
    F_y = C_aug_block @ S_z

    # 4. 构造权重矩阵块 [10,11](@ref)
    Q_big = np.kron(np.eye(N_pred), Q_cost)
    F_big = np.kron(np.eye(N_pred), F_cost)
    R_big = np.kron(np.eye(N_pred), R_cost)

    # 5. 构建控制输入增量到绝对输入的转换矩阵 [1](@ref)
    L_delta_u = np.zeros((N_pred * n_inputs, N_pred * n_inputs))
    for r_block in range(N_pred):
        for c_block in range(r_block + 1):
            row_start = r_block * n_inputs
            row_end = (r_block + 1) * n_inputs
            col_start = c_block * n_inputs
            col_end = (c_block + 1) * n_inputs
            L_delta_u[row_start:row_end, col_start:col_end] = np.eye(n_inputs)

    # 6. 构造QP问题的H矩阵和f向量 [10,11](@ref)
    H_qp = P_y.T @ Q_big @ P_y + F_big + L_delta_u.T @ R_big @ L_delta_u
    H_qp = 0.5 * (H_qp + H_qp.T)  # 确保对称性

    Z_current = np.vstack([current_X_koopman.reshape(-1, 1), prev_U.reshape(-1, 1)])
    Y_ref_vec = Y_ref_horizon.T.reshape(-1, 1)

    term_Q_related = P_y.T @ Q_big @ (F_y @ Z_current - Y_ref_vec)
    term_R_related = L_delta_u.T @ R_big @ np.kron(np.ones((N_pred, 1)), prev_U.reshape(-1, 1))
    f_qp = (term_Q_related + term_R_related).flatten()

    # 7. 构建不等式约束 [10,11](@ref)
    # 输入增量约束
    A_ineq_delta = np.vstack([np.eye(N_pred * n_inputs), -np.eye(N_pred * n_inputs)])
    b_ineq_delta = np.vstack([
        np.kron(np.ones((N_pred, 1)), max_abs_delta_U.reshape(-1, 1)),
        np.kron(np.ones((N_pred, 1)), max_abs_delta_U.reshape(-1, 1))
    ])

    # 绝对输入约束
    prev_U_horizon = np.kron(np.ones((N_pred, 1)), prev_U.reshape(-1, 1))
    A_ineq_abs_U_upper = L_delta_u
    b_ineq_abs_U_upper = np.kron(np.ones((N_pred, 1)), U_abs_max.reshape(-1, 1)) - prev_U_horizon

    A_ineq_abs_U_lower = -L_delta_u
    b_ineq_abs_U_lower = prev_U_horizon - np.kron(np.ones((N_pred, 1)), U_abs_min.reshape(-1, 1))

    # 合并所有约束
    A_ineq = np.vstack([A_ineq_delta, A_ineq_abs_U_upper, A_ineq_abs_U_lower])
    b_ineq = np.vstack([b_ineq_delta, b_ineq_abs_U_upper, b_ineq_abs_U_lower])

    # 8. 求解QP问题 [10,11](@ref)
    try:
        # 使用CVXOPT求解器 (更稳定)
        P = cvxopt.matrix(H_qp)
        q = cvxopt.matrix(f_qp)
        G = cvxopt.matrix(A_ineq)
        h = cvxopt.matrix(b_ineq)

        solution = cvxopt.solvers.qp(P, q, G, h, verbose=False)
        Delta_U_vec_optimal = np.array(solution['x']).flatten()
    except:
        # 回退到Scipy的SLSQP方法
        n_vars = N_pred * n_inputs
        bounds = [(-max_abs_delta_U[i % n_inputs], max_abs_delta_U[i % n_inputs])
                  for i in range(n_vars)]

        cons = [{
            'type': 'ineq',
            'fun': lambda x, A=A_ineq, b=b_ineq, i=i: b[i] - A[i, :] @ x
        } for i in range(A_ineq.shape[0])]

        res = minimize(
            fun=lambda x: 0.5 * x @ H_qp @ x + f_qp @ x,
            x0=np.zeros(n_vars),
            bounds=bounds,
            constraints=cons,
            method='SLSQP'
        )
        Delta_U_vec_optimal = res.x

    # 处理求解失败情况
    if Delta_U_vec_optimal is None:
        print('MPC: QP求解失败，返回零输入增量序列')
        return np.zeros((n_inputs, N_pred))

    # 重塑结果矩阵
    delta_U_optimal_sequence = Delta_U_vec_optimal.reshape(n_inputs, N_pred, order='F')
    return delta_U_optimal_sequence


if __name__ == '__main__':
    # 定义系统参数
    n_koopman_states = 4  # Koopman状态维度
    n_inputs = 2  # 控制输入数量
    n_outputs = 3  # 输出数量
    N_pred = 10  # 预测时域长度

    # 系统矩阵 (示例值)
    A_sys = np.array([
        [0.9, 0.1, 0.0, 0.0],
        [0.0, 0.8, 0.2, 0.0],
        [0.0, 0.0, 0.7, 0.3],
        [0.0, 0.0, 0.0, 0.6]
    ])
    B_sys = np.array([
        [0.5, 0.0],
        [0.3, 0.1],
        [0.2, 0.4],
        [0.0, 0.5]
    ])
    C_sys = np.array([
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 0.5, 0.5]
    ])

    # 权重矩阵
    Q_cost = np.diag([1.0, 0.8, 0.5])  # 输出误差权重
    F_cost = np.diag([0.1, 0.1])  # 输入增量权重
    R_cost = np.diag([0.01, 0.01])  # 输入幅值权重

    # 当前状态和上一时刻输入
    current_X_koopman = np.array([0.5, -0.2, 0.3, -0.1])
    prev_U = np.array([0.1, -0.05])

    # 参考轨迹 (从当前状态过渡到目标状态)
    Y_ref = np.array([1.5, 0.8, 0.0])  # 目标输出
    Y_ref_horizon = np.tile(Y_ref, (N_pred, 1)).T  # 预测时域内的参考轨迹

    # 约束条件
    max_abs_delta_U = np.array([0.2, 0.2])  # 输入增量约束
    U_abs_min = np.array([-0.8, -0.8])  # 输入下限
    U_abs_max = np.array([0.8, 0.8])  # 输入上限

    # ======================
    # 调用MPC控制器
    # ======================
    delta_U_seq = incremental_mpc(
        Q_cost, F_cost, R_cost, N_pred,
        A_sys, B_sys, C_sys,
        current_X_koopman, prev_U, Y_ref_horizon,
        max_abs_delta_U, U_abs_min, U_abs_max,
        n_inputs, n_outputs, n_koopman_states
    )

    # ======================
    # 可视化结果
    # ======================
    plt.figure(figsize=(12, 8))

    # 1. 绘制输入增量序列
    plt.subplot(2, 1, 1)
    for i in range(n_inputs):
        plt.step(range(N_pred), delta_U_seq[i], label=f'输入增量 {i + 1}')
    plt.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    plt.title('最优输入增量序列')
    plt.ylabel('增量值')
    plt.legend()
    plt.grid(True)

    # 2. 绘制绝对输入序列
    plt.subplot(2, 1, 2)
    U_seq = np.cumsum(delta_U_seq, axis=1) + prev_U.reshape(-1, 1)
    for i in range(n_inputs):
        plt.step(range(N_pred), U_seq[i], label=f'绝对输入 {i + 1}')
        plt.axhline(y=U_abs_max[i], color='r', linestyle='--', alpha=0.5)
        plt.axhline(y=U_abs_min[i], color='r', linestyle='--', alpha=0.5)
    plt.title('绝对输入序列')
    plt.xlabel('预测时域步长')
    plt.ylabel('输入值')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()

    # 打印第一控制步的结果
    print("最优输入增量序列 (第一控制步):")
    print(delta_U_seq[:, 0])