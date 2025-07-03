function delta_U_optimal_sequence = incrementalMPC(Q_cost, F_cost, R_cost, N_pred, ...
                                            A_sys, B_sys, C_sys, ...
                                            current_X_koopman, prev_U, Y_ref_horizon, ...
                                            max_abs_delta_U, U_abs_min, U_abs_max, ...
                                            n_inputs, n_outputs, n_koopman_states)
                                        
% incrementalMPC - 带有输入增量约束和绝对输入约束的MPC控制器
%
% 输入参数:
%   Q_cost (矩阵): 输出误差的权重矩阵 (n_outputs x n_outputs)
%   F_cost (矩阵): 输入增量的权重矩阵 (n_inputs x n_inputs)
%   R_cost (矩阵): 输入幅值的权重矩阵 (n_inputs x n_inputs)
%   N_pred (标量): 预测时域长度
%   A_sys (矩阵): Koopman状态转移矩阵A
%   B_sys (矩阵): Koopman输入矩阵B
%   C_sys (矩阵): Koopman输出矩阵C
%   current_X_koopman (向量): 当前Koopman状态
%   prev_U (向量): 上一时刻控制输入 (n_inputs x 1)
%   Y_ref_horizon (矩阵): 预测时域内的参考输出轨迹 (n_outputs x N_pred)
%   max_abs_delta_U (向量): 输入增量的最大绝对值限制 (n_inputs x 1)
%   U_abs_min (向量): 控制输入的最小绝对值限制 (n_inputs x 1)
%   U_abs_max (向量): 控制输入的最大绝对值限制 (n_inputs x 1)
%   n_inputs (标量): 控制输入数量
%   n_outputs (标量): 输出数量
%   n_koopman_states (标量): Koopman状态维度
%
% 输出参数:
%   delta_U_optimal_sequence (矩阵): 最优输入增量序列 (n_inputs x N_pred)
%                                    若求解器失败，返回零输入增量序列

    % 构建增广系统以处理输入增量
    % Z_k = [X_koopman_k; U_{k-1}]
    % Z_{k+1} = Aa * Z_k + Ba * delta_U_k
    % Y_k = Ca * Z_k
    
    Aa = [A_sys, B_sys; zeros(n_inputs, n_koopman_states), eye(n_inputs)];
    Ba = [B_sys; eye(n_inputs)];
    Ca = [C_sys, zeros(n_outputs, n_inputs)];

    n_aug_states = n_koopman_states + n_inputs;

    % QP问题构建：最小化 0.5 * Delta_U_vec^T * H * Delta_U_vec + f^T * Delta_U_vec
    % 约束条件：A_ineq * Delta_U_vec <= b_ineq

    % 创建预测矩阵 S_z 和 S_delta_u
    S_z = zeros(N_pred * n_aug_states, n_aug_states);
    S_delta_u = zeros(N_pred * n_aug_states, N_pred * n_inputs);
    
    temp_A_power_i = eye(n_aug_states);
    for i = 1:N_pred
        row_idx_aug = (i-1)*n_aug_states + 1 : i*n_aug_states;
        temp_A_power_i = temp_A_power_i * Aa; 
        S_z(row_idx_aug, :) = temp_A_power_i;      
        
        for j = 1:i
            col_idx_delta_u = (j-1)*n_inputs + 1 : j*n_inputs;
            power_Aa = eye(n_aug_states);
            if (i-j) > 0 % Aa^0 = 单位矩阵
                for p_count = 1:(i-j)
                    power_Aa = power_Aa * Aa;
                end
            end
            S_delta_u(row_idx_aug, col_idx_delta_u) = power_Aa * Ba;
        end
    end

    % 构建输出预测矩阵
    C_aug_block = kron(eye(N_pred), Ca);
    P_y = C_aug_block * S_delta_u;
    F_y = C_aug_block * S_z;
    
    % 构造权重矩阵块
    Q_big = kron(eye(N_pred), Q_cost);
    F_big = kron(eye(N_pred), F_cost);
    R_big = kron(eye(N_pred), R_cost);

    % 构建控制输入增量到绝对输入的转换矩阵
    % U_k = U_{k-1} + delta_U_k
    % U_{k+1} = U_{k-1} + delta_U_k + delta_U_{k+1}
    L_delta_u = zeros(N_pred * n_inputs, N_pred * n_inputs);
    for r_block = 1:N_pred % 行块（对应 U_0, U_1, ..., U_{N-1}）
        for c_block = 1:r_block % 列块（对应 DeltaU_0, DeltaU_1, ...）
            row_indices = (r_block-1)*n_inputs + (1:n_inputs);
            col_indices = (c_block-1)*n_inputs + (1:n_inputs);
            L_delta_u(row_indices, col_indices) = eye(n_inputs);
        end
    end
    
    % 构造QP问题的H矩阵
    H_qp = P_y' * Q_big * P_y + F_big + L_delta_u' * R_big * L_delta_u;
    H_qp = 0.5 * (H_qp + H_qp'); % 确保矩阵对称

    % 构造QP问题的f向量
    Z_current = [current_X_koopman; prev_U];
    Y_ref_vec = reshape(Y_ref_horizon, N_pred * n_outputs, 1);
    
    term_Q_related = P_y' * Q_big * (F_y * Z_current - Y_ref_vec);
    term_R_related = L_delta_u' * R_big * kron(ones(N_pred,1), prev_U);
    f_qp = term_Q_related + term_R_related;

    % --- 不等式约束构建 ---
    % 1. 输入增量约束：-max_abs_delta_U <= delta_U_i <= max_abs_delta_U
    A_ineq_delta = [eye(N_pred * n_inputs); -eye(N_pred * n_inputs)];
    b_ineq_delta = [kron(ones(N_pred, 1), max_abs_delta_U); kron(ones(N_pred, 1), max_abs_delta_U)];

    % 2. 绝对输入约束：U_abs_min <= U_i <= U_abs_max
    %    U_i = prev_U + sum_{j=0到i} delta_U_j
    %    U_序列 = kron(全1矩阵, prev_U) + L_delta_u * Delta_U_vec
    %    约束条件分解为上下界
    
    % 上界约束：L_delta_u * Delta_U_vec <= U_abs_max_horizon - prev_U_horizon
    A_ineq_abs_U_upper = L_delta_u;
    b_ineq_abs_U_upper = kron(ones(N_pred,1), U_abs_max) - kron(ones(N_pred,1), prev_U);
    
    % 下界约束：-L_delta_u * Delta_U_vec <= prev_U_horizon - U_abs_min_horizon
    A_ineq_abs_U_lower = -L_delta_u;
    b_ineq_abs_U_lower = kron(ones(N_pred,1), prev_U) - kron(ones(N_pred,1), U_abs_min);

    % 合并所有不等式约束
    A_ineq = [A_ineq_delta; A_ineq_abs_U_upper; A_ineq_abs_U_lower];
    b_ineq = [b_ineq_delta; b_ineq_abs_U_upper; b_ineq_abs_U_lower];

    % 求解QP问题
    options = optimoptions('quadprog','Display','none');
    Delta_U_vec_optimal = quadprog(H_qp, f_qp, A_ineq, b_ineq, [], [], [], [], [], options);

    if isempty(Delta_U_vec_optimal)
        warning('MPC: quadprog求解失败，返回零输入增量序列');
        delta_U_optimal_sequence = zeros(n_inputs, N_pred);
    else
        delta_U_optimal_sequence = reshape(Delta_U_vec_optimal, n_inputs, N_pred);
    end
end