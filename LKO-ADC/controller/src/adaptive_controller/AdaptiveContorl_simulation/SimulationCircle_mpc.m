%% ------------------------------------------------------------------------
%                     文件头和路径设置
% -------------------------------------------------------------------------
% 获取当前文件所在目录
currentDir = fileparts(mfilename('fullpath'));
% 获取上一级目录
parentDir = fileparts(currentDir);
% 只添加上一级目录本身（不包括其子目录）
addpath(parentDir);
%% ------------------------------------------------------------------------
%                           初始化
% -------------------------------------------------------------------------
%% 清屏
clear;
close all;
clc;
%% 基本参数
k_steps = 200; % 总仿真步数
n_states_original = 6; % 原始状态维度 (x_orig)

% ========================= 参数更新的超参数 =========================
window_size = 10; % 滑动窗口的大小 (N)

% --- 学习率 (非常重要，需要仔细调试！) ---
% --- 建议从 1e-7 ~ 1e-9 的范围开始尝试 ---
eta_A = 4e-12;  % A_bar 的学习率 (已修改)
eta_B = 1e-3;  % B_bar 的学习率 (已修改，必须为非零)

%% 加载Koopman算子
% --- Koopman和提升函数 ---
delay_time = 7;
target_dimensions = 24;
lift_function = @polynomial_expansion_td; 
trajectory_function = @generateReferenceLemniscate;
km_path = '../koopman_model/poly_delay7_lift24.mat'; % 确保这是您的实际路径
koopman_parmas = load(km_path);

% 真实的、不变的系统模型 (模拟现实世界)
A_true = koopman_parmas.A; 
B_true = koopman_parmas.B; 

% MPC内部使用的、存在初始误差且需要被自适应更新的模型
A_bar = A_true - 0.03*A_true; 
B_bar = B_true - 0.03*B_true; 

% 存储初始模型用于对比
A_bar_init = A_bar;
B_bar_init = B_bar;

n_StateEigen = size(A_true,1); % Koopman 状态维度 (提升后的维度)
n_InputEigen = size(B_true,2); % 输入维度 (U的维度)
n_Output = 6;                  % 输出维度 (Y的维度)
C = [eye(n_Output,n_Output), zeros(n_Output, n_StateEigen - n_Output)];

%% --- 控制输入约束 ---
maxIncremental = 0.1; 
U_abs_min = [0;0;0;0;0;0];
U_abs_max = [5;5;5;5;5;5];

%% 生成参考圆轨迹
initialState_original = [14.23;-4.42;-215.35;18.95;-22.45;-339.89];
initialState_original = repmat(initialState_original, delay_time, 1);
R_circle2 = 45; R_circle1 = 6;
weights = linspace(0,1,100);
center1 = initialState_original(1:3);
Y_ref1 = trajectory_function(center1, R_circle1, k_steps);
to_center1 = center1*(1-weights)+Y_ref1(:,1)*weights;
Y_ref1 = [to_center1, Y_ref1, repmat(Y_ref1(:, end), 1, 20)];
center2 = initialState_original(4:6);
Y_ref2 = trajectory_function(center2, R_circle2, k_steps);
to_center2 = center2*(1-weights)+Y_ref2(:,1)*weights;
Y_ref2 = [to_center2, Y_ref2, repmat(Y_ref2(:, end), 1, 20)];
Y_ref = [Y_ref1;Y_ref2];
k_steps = k_steps + 100;

%% MPC控制器参数定义
Q_cost_diag = [10, 10, 10, 10, 10, 10];
Q_cost = diag(Q_cost_diag); 
F_cost_val = 0.1; 
F_cost = eye(n_InputEigen) * F_cost_val;
R_cost_val = 0.01;
R_cost = eye(n_InputEigen) * R_cost_val;
N_pred = 10;
max_abs_delta_U = ones(n_InputEigen, 1) * maxIncremental;

%% MPC仿真循环
% --- 初始化仿真变量 ---
X_koopman_current = lift_function(initialState_original, target_dimensions, delay_time);

% 历史数据用于绘图
X_koopman_history = zeros(n_StateEigen, k_steps + 1);
Y_history = zeros(n_Output, k_steps + 1);
U_history = zeros(n_InputEigen, k_steps);

X_koopman_history(:,1) = X_koopman_current;
Y_history(:,1) = C * X_koopman_current;
prev_U = 2*ones(n_InputEigen, 1);

% ==================== 初始化用于更新的历史数据窗口 (已修改) ====================
X_win_hist = zeros(n_StateEigen, window_size);
U_win_hist = zeros(n_InputEigen, window_size);
X_next_win_hist = zeros(n_StateEigen, window_size); % 用于存储真实的 x_{k+1}
% ==========================================================================

fprintf('Starting MPC simulation for %d steps...\n', k_steps);
for k = 1:k_steps
    if mod(k, 50) == 0
        fprintf('Simulation step: %d/%d\n', k, k_steps);
    end
    
    % 提取当前视界的参考轨迹
    if k + N_pred -1 <= size(Y_ref, 2)
        Y_ref_horizon = Y_ref(:, k : k + N_pred - 1);
    else
        num_remaining_ref = size(Y_ref, 2) - k + 1;
        Y_ref_horizon_temp = Y_ref(:, k : end);
        Y_ref_horizon = [Y_ref_horizon_temp, ...
                         repmat(Y_ref(:, end), 1, N_pred - num_remaining_ref)];
    end
    
    % 调用MPC控制器获取最优控制输入增量序列
    delta_U_optimal_sequence = incrementalMPC(...
                                Q_cost, F_cost, R_cost, N_pred, ...
                                A_bar, B_bar, C, ... % 使用需要更新的 A_bar, B_bar
                                X_koopman_current, prev_U, Y_ref_horizon, ...
                                max_abs_delta_U, U_abs_min, U_abs_max, ...
                                n_InputEigen, n_Output, n_StateEigen);
    current_delta_U = delta_U_optimal_sequence(:, 1);
    current_U = prev_U + current_delta_U;
    
    % 更新系统状态 (使用真实的、物理世界的Koopman模型 A_true, B_true)
    X_koopman_next = A_true * X_koopman_current + B_true * current_U;
    
    % 计算系统输出
    Y_next = C * X_koopman_next;
    
    % 存储历史数据用于绘图
    U_history(:, k) = current_U;
    X_koopman_history(:, k+1) = X_koopman_next;
    Y_history(:, k+1) = Y_next;
    
    % ======================= 在线更新模型部分 (已修改) =======================
    % 1. 更新滑动窗口数据库 (使用 k 时刻的数据)
    X_win_hist = [X_win_hist(:, 2:end), X_koopman_current]; % 存储 x_k
    U_win_hist = [U_win_hist(:, 2:end), current_U];         % 存储 u_k
    X_next_win_hist = [X_next_win_hist(:, 2:end), X_koopman_next]; % 存储真实的 x_{k+1}

    % 2. 当窗口数据填满后，开始执行更新
    %    (修改为每一步都更新，以提高适应速度)
    if k >= window_size
        % 调用更新函数计算梯度
        [grad_A, grad_B] = update_AB_sliding_window(A_bar, B_bar, ...
                                              X_win_hist, U_win_hist, X_next_win_hist);
        
        % 应用梯度下降更新模型
        A_bar = A_bar - eta_A * grad_A;
        B_bar = B_bar - eta_B * grad_B;
        
        % (可选) 增加调试信息，监控更新幅度
        if mod(k, 20) == 0 % 每20步打印一次
            update_norm_A = norm(eta_A * grad_A, 'fro');
            update_norm_B = norm(eta_B * grad_B, 'fro');
            fprintf('Step %d: Norm of A_update=%.2e, Norm of B_update=%.2e\n', k, update_norm_A, update_norm_B);
        end
    end
    % ====================================================================
    
    % 更新状态和前一个输入以进行下一次循环
    X_koopman_current = X_koopman_next;
    prev_U = current_U;
end

A_d = A_bar - A_bar_init;
B_d = B_bar - B_bar_init;
fprintf('MPC simulation finished.\n');
fprintf('Total change in A_bar (Frobenius norm): %f\n', norm(A_d, 'fro'));
fprintf('Total change in B_bar (Frobenius norm): %f\n', norm(B_d, 'fro'));

%% 结果计算与绘制
mse1 = calculateRMSE(Y_ref(1:3, 1:k_steps+1), Y_history(1:3,:));
mse2 = calculateRMSE(Y_ref(4:6, 1:k_steps+1), Y_history(4:6,:));
fprintf('第一关节轨迹跟踪均方根误差为: %f, 第二关节轨迹跟踪的均方根误差为: %f\n', mse1, mse2);

% --- 绘图部分 ---
figure;
plot3(Y_history(1,:), Y_history(2,:), Y_history(3,:), 'b-', 'LineWidth', 1.5);
hold on;
plot3(Y_ref(1,1:k_steps+1), Y_ref(2,1:k_steps+1), Y_ref(3,1:k_steps+1), 'r--', 'LineWidth', 1.5);
plot3(Y_history(1,1), Y_history(2,1), Y_history(3,1), 'go', 'MarkerSize', 10, 'MarkerFaceColor', 'g'); % 起点
plot3(Y_ref(1,1), Y_ref(2,1), Y_ref(3,1), 'ro', 'MarkerSize', 10, 'MarkerFaceColor', 'r'); % 参考起点
xlabel('X position');
ylabel('Y position');
zlabel('Z position');
title('3D Trajectory Tracking - Joint Group 1');
axis equal;
grid on;
legend('Actual Trajectory', 'Reference Trajectory', 'Actual Start', 'Reference Start');
view(3);

figure;
plot3(Y_history(4,:), Y_history(5,:), Y_history(6,:), 'b-', 'LineWidth', 1.5);
hold on;
plot3(Y_ref(4,1:k_steps+1), Y_ref(5,1:k_steps+1), Y_ref(6,1:k_steps+1), 'r--', 'LineWidth', 1.5);
xlabel('X position');
ylabel('Y position');
zlabel('Z position');
% title('Normal delayK-MPC Simulation', 'Position', [0, 8, 0]);
axis equal;
grid on;
legend('Actual Trajectory', 'Reference Trajectory', 'Location','north','NumColumns',2);
view(3);

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