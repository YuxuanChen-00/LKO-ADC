%% ========================================================================
%                     滑动窗口更新函数 (已修改)
% =========================================================================
function [grad_A_avg, grad_B_avg] = update_AB_sliding_window(A_current, B_current, ...
                                                   X_window, U_window, X_next_window)
% 使用滑动窗口内的数据，通过梯度下降法计算A和B矩阵的梯度
%
% 输入:
%   A_current, B_current: 当前的A, B矩阵估计值
%   X_window:      状态历史窗口, 维度为 (n_states x window_size), 包含 x_k
%   U_window:      输入历史窗口, 维度为 (n_inputs x window_size), 包含 u_k
%   X_next_window: 真实下一状态历史窗口, 维度为 (n_states x window_size), 包含 x_{k+1}
%
% 输出:
%   grad_A_avg, grad_B_avg: 计算出的A和B的平均梯度

    % 窗口长度
    N_window = size(X_window, 2);
    
    % 初始化梯度累加器
    grad_A_sum = zeros(size(A_current));
    grad_B_sum = zeros(size(B_current));
    
    % 遍历窗口内的数据点来累加梯度
    for i = 1:N_window
        % 提取数据点 (x_i, u_i) 和真实的下一状态 x_{i+1}
        x_i = X_window(:, i);
        u_i = U_window(:, i);
        x_i_plus_1_real = X_next_window(:, i); % <-- 使用正确的真实下一时刻状态

        % 使用当前模型进行预测
        x_i_plus_1_pred = A_current * x_i + B_current * u_i;
        
        % 计算状态预测误差
        error_vec = x_i_plus_1_pred - x_i_plus_1_real; % 注意：梯度推导中的误差是 pred-real 
                                                       % 所以更新时用 A = A - eta * grad，即 A = A - eta * (pred-real)*x'
                                                       % 如果误差定义为 real-pred, 更新法则是 A = A + eta*grad
        
        % 计算损失函数对A和B的梯度: L = 1/2 * ||error||^2
        % dL/dA = dL/de * de/dA = error * x_i'
        grad_A_single = error_vec * x_i';
        
        % dL/dB = dL/de * de/dB = error * u_i'
        grad_B_single = error_vec * u_i';
        
        % 累加梯度
        grad_A_sum = grad_A_sum + grad_A_single;
        grad_B_sum = grad_B_sum + grad_B_single;
    end
    
    % 对梯度求平均，使学习率不严重依赖于窗口大小
    grad_A_avg = grad_A_sum / N_window;
    grad_B_avg = grad_B_sum / N_window;
end