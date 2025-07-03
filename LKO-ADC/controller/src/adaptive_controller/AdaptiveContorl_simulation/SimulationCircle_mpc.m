%% ------------------------------------------------------------------------
%                     文件头和路径设置 (您的原始代码)
% -------------------------------------------------------------------------
% 获取当前文件所在目录
currentDir = fileparts(mfilename('fullpath'));
% 获取上一级目录
parentDir = fileparts(currentDir);
% 只添加上一级目录本身（不包括其子目录）
addpath(parentDir);

%% ------------------------------------------------------------------------
%                           初始化 (您的原始代码)
% -------------------------------------------------------------------------
%% 清屏
% clear;
% close all;
% clc;

%% 基本参数
k_steps = 200; % 总仿真步数
n_states_original = 6; % 原始状态维度 (x_orig)

% ========================= 新增：参数更新的超参数 =========================
window_size = 10; % 滑动窗口的大小 (N)
% --- 学习率 (非常重要，需要仔细调试！) ---
% --- 从非常小的值开始，如 1e-7, 1e-8, ... ---
eta_A = 6.8e-11; % A_bar 的学习率
eta_B = 0e-11; % B_bar 的学习率
% =======================================================================

%% 加载Koopman算子 (您的原始代码)
% --- Koopman和提升函数 ---
delay_time = 7;
target_dimensions = 24;
lift_function = @polynomial_expansion_td; 
km_path = '../koopman_model/poly_delay7_lift24.mat'; % 修改为您的实际路径
koopman_parmas = load(km_path);
A = koopman_parmas.A; % Koopman 状态转移矩阵 (n_StateEigen x n_StateEigen)
B = koopman_parmas.B; % Koopman 输入矩阵 (n_StateEigen x n_InputEigen)

A_bar = A - 0.05*A; % MPC内部的、需要被更新的模型
B_bar = B - 0.00*B; % MPC内部的、需要被更新的模型
A_bar_init = A_bar;
B_bar_init = B_bar;

n_StateEigen = size(A,1); % Koopman 状态维度 (提升后的维度)
n_InputEigen = size(B,2); % 输入维度 (U的维度)
n_Output = 6;             % 输出维度 (Y的维度)
C = [eye(n_Output,n_Output), zeros(n_Output, n_StateEigen - n_Output)];

%% --- 控制输入约束 --- (您的原始代码)
maxIncremental = 0.1; 
U_abs_min = [0;0;0;0;0;0];
U_abs_max = [5;5;5;5;5;5];

%% 生成参考圆轨迹 (您的原始代码)
initialState_original = [14.23;-4.42;-215.35;18.95;-22.45;-339.89];
initialState_original = repmat(initialState_original, delay_time, 1);
R_circle2 = 45; R_circle1 = 6;
weights = linspace(0,1,100);
center1 = initialState_original(1:3);
Y_ref1 = generateReferenceCircle(center1, R_circle1, k_steps);
to_center1 = center1*(1-weights)+Y_ref1(:,1)*weights;
Y_ref1 = [to_center1, Y_ref1, repmat(Y_ref1(:, end), 1, 20)];
center2 = initialState_original(4:6);
Y_ref2 = generateReferenceCircle(center2, R_circle2, k_steps);
to_center2 = center2*(1-weights)+Y_ref2(:,1)*weights;
Y_ref2 = [to_center2, Y_ref2, repmat(Y_ref2(:, end), 1, 20)];
Y_ref = [Y_ref1;Y_ref2];
k_steps = k_steps + 100;

%% MPC控制器参数定义 (您的原始代码)
Q_cost_diag = [10, 10, 10, 10, 10, 10];
Q_cost = diag(Q_cost_diag); 
F_cost_val = 0.1; 
F_cost = eye(n_InputEigen) * F_cost_val;
R_cost_val = 0.01;
R_cost = eye(n_InputEigen) * R_cost_val;
N_pred = 10;
max_abs_delta_U = ones(n_InputEigen, 1) * maxIncremental;

%% MPC仿真循环
% --- 初始化仿真变量 --- (部分修改)
X_koopman_current = lift_function(initialState_original, target_dimensions, delay_time);

% 历史数据用于绘图
X_koopman_history = zeros(n_StateEigen, k_steps + 1);
Y_history = zeros(n_Output, k_steps + 1);
U_history = zeros(n_InputEigen, k_steps);

X_koopman_history(:,1) = X_koopman_current;
Y_history(:,1) = C * X_koopman_current;
prev_U = 2*ones(n_InputEigen, 1);

% ==================== 新增：初始化用于更新的历史数据窗口 ====================
X_win_hist = zeros(n_StateEigen, window_size);
U_win_hist = zeros(n_InputEigen, window_size);
Y_win_hist = zeros(n_Output, window_size);
% ==========================================================================

fprintf('Starting MPC simulation for %d steps...\n', k_steps);

for k = 1:k_steps
    if mod(k, 50) == 0
        fprintf('Simulation step: %d/%d\n', k, k_steps);
    end
    
    % 提取当前视界的参考轨迹 (您的原始代码)
    if k + N_pred -1 <= size(Y_ref, 2)
        Y_ref_horizon = Y_ref(:, k : k + N_pred - 1);
    else
        num_remaining_ref = size(Y_ref, 2) - k + 1;
        Y_ref_horizon_temp = Y_ref(:, k : end);
        Y_ref_horizon = [Y_ref_horizon_temp, ...
                         repmat(Y_ref(:, end), 1, N_pred - num_remaining_ref)];
    end
    
    % 调用MPC控制器获取最优控制输入增量序列 (您的原始代码)
    % delta_U_optimal_sequence = incrementalMPC(...
    %                             Q_cost, F_cost, R_cost, N_pred, ...
    %                             A_bar, B_bar, C, ... % 使用需要更新的 A_bar, B_bar
    %                             X_koopman_current, prev_U, Y_ref_horizon, ...
    %                             max_abs_delta_U, U_abs_min, U_abs_max, ...
    %                             n_InputEigen, n_Output, n_StateEigen);
    % 
    % current_delta_U = delta_U_optimal_sequence(:, 1);
    % current_U = prev_U + current_delta_U;
    
    current_U = pinv(C*B_bar)*(Y_ref(:, k) + C*A_bar*X_koopman_current);
    current_U = min(max(current_U, U_abs_min), U_abs_max);
    current_U = [2;2;2;2;2;2];
    

    % 更新系统状态 (使用真实的Koopman模型 A, B)
    X_koopman_next = A * X_koopman_current + B * current_U;
    
    % 计算系统输出
    Y_next = C * X_koopman_next;
    
    % 存储历史数据用于绘图
    U_history(:, k) = current_U;
    X_koopman_history(:, k+1) = X_koopman_next;
    Y_history(:, k+1) = Y_next;
    
    % ======================= 新增：在线更新模型部分 =======================
    % 1. 更新滑动窗口数据库 (使用 k 时刻的数据)
    %    X_koopman_current 对应 x_i
    %    current_U 对应 u_i
    %    Y_next 对应 y_{i+1}
    if k > 1 % 从第二个点开始有完整的 (x,u,y_next)
        X_win_hist = [X_win_hist(:, 2:end), X_koopman_current];
        U_win_hist = [U_win_hist(:, 2:end), current_U];
        Y_win_hist = [Y_win_hist(:, 2:end), Y_next];
    end

    % 2. 当窗口数据填满后，开始执行更新
    if k >= window_size && mod(k, window_size) == 0
        % 调用更新函数
        [delta_A, delta_B] = update_AB_sliding_window(A_bar, B_bar, C, ...
                                                  X_win_hist, U_win_hist, Y_win_hist, ...
                                                  eta_A, eta_B);
        A_bar = A_bar - eta_A*delta_A;
        B_bar = B_bar - eta_B*delta_B;
    end
    % ====================================================================

    % 更新状态和前一个输入以进行下一次循环
    X_koopman_current = X_koopman_next;
    prev_U = current_U;
end
A_d = A_bar - A_bar_init;
B_d = B_bar - B_bar_init;
fprintf('MPC simulation finished.\n');

%% 结果计算与绘制 (您的原始代码)
mse1 = calculateRMSE(Y_ref(1:3, 1:k_steps+1), Y_history(1:3,:));
mse2 = calculateRMSE(Y_ref(4:6, 1:k_steps+1), Y_history(4:6,:));
fprintf('第一关节轨迹跟踪均方根误差为: %2f, 第二关节轨迹跟踪的均方根误差为: %2f\n', mse1, mse2);

% --- 绘图部分省略，与您的原始代码相同 ---
% ...
figure;
plot3(Y_history(1,:), Y_history(2,:), Y_history(3,:), 'b-', 'LineWidth', 1.5);
hold on;
plot3(Y_ref(1,1:k_steps+1), Y_ref(2,1:k_steps+1), Y_ref(3,1:k_steps+1), 'r--', 'LineWidth', 1.5);
plot3(Y_history(1,1), Y_history(2,1), Y_history(3,1), 'go', 'MarkerSize', 10, 'MarkerFaceColor', 'g'); % 起点
plot3(Y_ref(1,1), Y_ref(2,1), Y_ref(3,1), 'ro', 'MarkerSize', 10, 'MarkerFaceColor', 'r'); % 参考起点
xlabel('X position');
ylabel('Y position');
zlabel('Z position');
title('3D Trajectory Tracking');
axis equal;
grid on;
plot3(Y_history(4,:), Y_history(5,:), Y_history(6,:), 'b-', 'LineWidth', 1.5);
hold on;
plot3(Y_ref(4,1:k_steps+1), Y_ref(5,1:k_steps+1), Y_ref(6,1:k_steps+1), 'r--', 'LineWidth', 1.5);
plot3(Y_history(4,1), Y_history(5,1), Y_history(6,1), 'go', 'MarkerSize', 10, 'MarkerFaceColor', 'g'); % 起点
plot3(Y_ref(4,1), Y_ref(5,1), Y_ref(6,1), 'ro', 'MarkerSize', 10, 'MarkerFaceColor', 'r'); % 参考起点
xlabel('X position');
ylabel('Y position');
zlabel('Z position');
title('3D Trajectory Tracking');
legend('Actual Trajectory1', 'Reference Trajectory1', 'Actual Start1', 'Reference Start1', ...
    'Actual Trajectory2', 'Reference Trajectory2', 'Actual Start2', 'Reference Start2');
axis equal;
grid on;
view(3);

%% ========================================================================
%                     新增：滑动窗口更新函数
% =========================================================================
function [grad_A_avg, grad_B_avg] = update_AB_sliding_window(A_current, B_current, C_output, ...
                                                   X_window, U_window, Y_window, ...
                                                   eta_A, eta_B)
% 使用滑动窗口内的数据，通过梯度下降法更新A和B矩阵
%
% 输入:
%   A_current, B_current: 当前的A, B矩阵
%   C_output: 输出矩阵 C (y = C*x)
%   X_window: 状态历史窗口, 维度为 (n_states x window_size)
%   U_window: 输入历史窗口, 维度为 (n_inputs x window_size)
%   Y_window: 输出历史窗口, 维度为 (n_outputs x window_size)
%   eta_A, eta_B: 学习率
%
% 输出:
%   A_new, B_new: 更新后的A, B矩阵

    % 窗口长度
    N_window = size(X_window, 2);

    % 初始化梯度累加器
    grad_A_sum = zeros(size(A_current));
    grad_B_sum = zeros(size(B_current));

    % 遍历窗口内的数据点来累加梯度
    % 窗口数据为 (x_i, u_i) -> y_{i+1}，所以循环到 N-1
    % 注意：这里的 X_window 存储的是 x_0, x_1, ..., x_{N-1}
    %       U_window 存储的是 u_0, u_1, ..., u_{N-1}
    %       Y_window 存储的是 y_1, y_2, ..., y_N
    for i = 1:N_window
        % 提取数据点
        x_i = X_window(:, i);
        u_i = U_window(:, i);
        x_i_plus_1_real = X_window(:, i);
        y_i_plus_1_real = Y_window(:, i);

        % 使用当前模型进行预测
        y_i_plus_1_pred = C_output * (A_current * x_i + B_current * u_i);
        x_i_plus_1_pred = A_current * x_i + B_current * u_i;

        % 计算输出误差
        % error_vec = y_i_plus_1_real - y_i_plus_1_pred;
        error_vec = x_i_plus_1_real - x_i_plus_1_pred;

        % % 计算单个数据点的梯度 (基于输出误差 y)
        % % grad_J_A = -C' * error * x'
        % grad_A_single = -C_output' * error_vec * x_i';
        % % grad_J_B = -C' * error * u'
        % grad_B_single = -C_output' * error_vec * u_i';

        % 计算单个数据点的梯度 (基于输出误差 x)
        % grad_J_A = -C' * error * x'
        grad_A_single = -error_vec * x_i';
        % grad_J_B = -C' * error * u'
        grad_B_single = -error_vec * u_i';

        % 累加梯度
        grad_A_sum = grad_A_sum + grad_A_single;
        grad_B_sum = grad_B_sum + grad_B_single;
        % disp(['A的梯度是:' num2str(sum(abs(grad_A_single))) '  B的梯度是:' num2str(sum(abs(grad_B_single)))])
    end
    
    % 对梯度求平均，使学习率不依赖于窗口大小
    grad_A_avg = grad_A_sum / N_window;
    grad_B_avg = grad_B_sum / N_window;
    
    % 应用梯度下降更新法则
    % 新参数 = 旧参数 - 学习率 * 梯度
    A_new = A_current - eta_A * grad_A_avg;
    B_new = B_current - eta_B * grad_B_avg;
end