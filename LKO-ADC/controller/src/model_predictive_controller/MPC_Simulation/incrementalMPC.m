function delta_U_optimal_sequence = incrementalMPC(Q_cost, F_cost, R_cost, N_pred, ...
                                            A_sys, B_sys, C_sys, ...
                                            current_X_koopman, prev_U, Y_ref_horizon, ...
                                            max_abs_delta_U, U_abs_min, U_abs_max, ...
                                            n_inputs, n_outputs, n_koopman_states)
% mpcControllerFullConstraints - MPC controller with input increment and absolute input constraints.
%
% Inputs:
%   Q_cost (matrix): Weight matrix for output error (n_outputs x n_outputs).
%   F_cost (matrix): Weight matrix for input increment (n_inputs x n_inputs).
%   R_cost (matrix): Weight matrix for input magnitude (n_inputs x n_inputs).
%   N_pred (scalar): Prediction horizon.
%   A_sys (matrix): Koopman state transition matrix A.
%   B_sys (matrix): Koopman input matrix B.
%   C_sys (matrix): Koopman output matrix C.
%   current_X_koopman (vector): Current Koopman state.
%   prev_U (vector): Previous control input (n_inputs x 1).
%   Y_ref_horizon (matrix): Reference output trajectory for the horizon (n_outputs x N_pred).
%   max_abs_delta_U (vector): Maximum absolute value for input increments (n_inputs x 1).
%   U_abs_min (vector): Minimum absolute value for control inputs U (n_inputs x 1).
%   U_abs_max (vector): Maximum absolute value for control inputs U (n_inputs x 1).
%   n_inputs (scalar): Number of control inputs.
%   n_outputs (scalar): Number of outputs.
%   n_koopman_states (scalar): Number of Koopman states.
%
% Outputs:
%   delta_U_optimal_sequence (matrix): Optimal sequence of input increments (n_inputs x N_pred).
%                                      Returns zeros if solver fails.

    % Augmented system for input increment formulation
    % Z_k = [X_koopman_k; U_{k-1}]
    % Z_{k+1} = Aa * Z_k + Ba * delta_U_k
    % Y_k = Ca * Z_k
    
    Aa = [A_sys, B_sys; zeros(n_inputs, n_koopman_states), eye(n_inputs)];
    Ba = [B_sys; eye(n_inputs)];
    Ca = [C_sys, zeros(n_outputs, n_inputs)];

    n_aug_states = n_koopman_states + n_inputs;

    % QP formulation: min 0.5 * Delta_U_vec^T * H * Delta_U_vec + f^T * Delta_U_vec
    % Subject to: A_ineq * Delta_U_vec <= b_ineq

    % Create prediction matrices S_z and S_delta_u
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
            if (i-j) > 0 % Aa^0 = I
                for p_count = 1:(i-j)
                    power_Aa = power_Aa * Aa;
                end
            end
            S_delta_u(row_idx_aug, col_idx_delta_u) = power_Aa * Ba;
        end
    end

    C_aug_block = kron(eye(N_pred), Ca);
    P_y = C_aug_block * S_delta_u;
    F_y = C_aug_block * S_z;
    
    Q_big = kron(eye(N_pred), Q_cost);
    F_big = kron(eye(N_pred), F_cost);
    R_big = kron(eye(N_pred), R_cost);

    % Construct L_delta_u matrix for U_sequence = L_u_prev * U_prev + L_delta_u * Delta_U_vec
    % U_k = U_{k-1} + delta_U_k
    % U_{k+1} = U_{k-1} + delta_U_k + delta_U_{k+1}
    L_delta_u = zeros(N_pred * n_inputs, N_pred * n_inputs);
    for r_block = 1:N_pred % row blocks (for U_0, U_1, ..., U_{N-1})
        for c_block = 1:r_block % col blocks (for DeltaU_0, DeltaU_1, ...)
            row_indices = (r_block-1)*n_inputs + (1:n_inputs);
            col_indices = (c_block-1)*n_inputs + (1:n_inputs);
            L_delta_u(row_indices, col_indices) = eye(n_inputs);
        end
    end
    
    % H matrix for QP
    H_qp = P_y' * Q_big * P_y + F_big + L_delta_u' * R_big * L_delta_u;
    H_qp = 0.5 * (H_qp + H_qp'); % Ensure symmetry

    % f vector for QP
    Z_current = [current_X_koopman; prev_U];
    Y_ref_vec = reshape(Y_ref_horizon, N_pred * n_outputs, 1);
    
    term_Q_related = P_y' * Q_big * (F_y * Z_current - Y_ref_vec);
    term_R_related = L_delta_u' * R_big * kron(ones(N_pred,1), prev_U);
    f_qp = term_Q_related + term_R_related;

    % --- Inequality constraints ---
    % 1. Input increment constraints: -max_abs_delta_U <= delta_U_i <= max_abs_delta_U
    A_ineq_delta = [eye(N_pred * n_inputs); -eye(N_pred * n_inputs)];
    b_ineq_delta = [kron(ones(N_pred, 1), max_abs_delta_U); kron(ones(N_pred, 1), max_abs_delta_U)];

    % 2. Absolute input constraints: U_abs_min <= U_i <= U_abs_max
    %    U_i = prev_U + sum_{j=0 to i} delta_U_j
    %    U_sequence = kron(ones(N_pred,1), prev_U) + L_delta_u * Delta_U_vec
    %    So, U_abs_min <= kron(ones(N_pred,1), prev_U) + L_delta_u * Delta_U_vec <= U_abs_max
    
    % L_delta_u * Delta_U_vec <= U_abs_max_horizon - prev_U_horizon
    A_ineq_abs_U_upper = L_delta_u;
    b_ineq_abs_U_upper = kron(ones(N_pred,1), U_abs_max) - kron(ones(N_pred,1), prev_U);
    
    % -L_delta_u * Delta_U_vec <= prev_U_horizon - U_abs_min_horizon
    A_ineq_abs_U_lower = -L_delta_u;
    b_ineq_abs_U_lower = kron(ones(N_pred,1), prev_U) - kron(ones(N_pred,1), U_abs_min);

    % Combine all inequality constraints
    A_ineq = [A_ineq_delta; A_ineq_abs_U_upper; A_ineq_abs_U_lower];
    b_ineq = [b_ineq_delta; b_ineq_abs_U_upper; b_ineq_abs_U_lower];

    % Solve QP
    options = optimoptions('quadprog', 'Display', 'none', 'Algorithm','interior-point-convex'); % 'active-set' can also be used
    Delta_U_vec_optimal = quadprog(H_qp, f_qp, A_ineq, b_ineq, [], [], [], [], [], options);

    if isempty(Delta_U_vec_optimal)
        warning('MPC: quadprog failed to find a solution. Returning zero input increments.');
        delta_U_optimal_sequence = zeros(n_inputs, N_pred);
    else
        delta_U_optimal_sequence = reshape(Delta_U_vec_optimal, n_inputs, N_pred);
    end
end