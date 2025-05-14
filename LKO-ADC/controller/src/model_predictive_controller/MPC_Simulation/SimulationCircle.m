%% 将主文件夹添加到目录
addpath(genpath('F:\2 软体机器人建模与控制\Bellow-Koopman')); % 修改为您的实际路径
%% 清屏
clear;
close all;
clc;
%% 基本参数
k_steps = 300; % 总仿真步数
n_states_original = 12; % 原始状态维度 (x_orig)

%% 加载Koopman算子
% --- Koopman和提升函数 ---
lift_function = @lift_polynomial; 
km_path = '../../../data/SorotokiPoly/delay3_lift64_F0.04.mat'; % 修改为您的实际路径
koopman_parmas = load(km_path);
A = koopman_parmas.A; % Koopman 状态转移矩阵 (n_StateEigen x n_StateEigen)
B = koopman_parmas.B; % Koopman 输入矩阵 (n_StateEigen x n_InputEigen)
params_control = koopman_parmas.params_control;
params_state = koopman_parmas.params_state;

n_StateEigen = size(A,1); % Koopman 状态维度 (提升后的维度)
n_InputEigen = size(B,2); % 输入维度 (U的维度)
n_Output = 6;             % 输出维度 (Y的维度)

% 输出矩阵C: 从Koopman状态提取输出Y
% 假设原始状态的前12个维度被包含在提升后的状态中，
% 并且Y对应于原始状态的第7到12个维度，即提升后状态的第7到12个维度。
if n_StateEigen < n_states_original
    error('n_StateEigen must be >= n_states_original for the given C matrix structure.');
end
C = [zeros(n_Output,6), eye(n_Output), zeros(n_Output, n_StateEigen - n_states_original)];

%% --- 控制输入约束 ---
maxIncremental = 0.1; % 最大控制输入增量 (标量，假设对所有输入相同)
U_abs_min = [0;0;0;0;0;0];
U_abs_max = [1;1;1;1;1;1];

%% 生成参考圆轨迹
R_circle = 20; % 圆半径
% 初始原始状态 (12维)
initialState_original = [25.19, -5.34,-195.82,1,1,1,33.43,-4.93,-290.35,1,1,1]';
initialState_original_norm = normalize_data(initialState_original, params_state);
direction_theta = 1; % 圆方向

% 为MPC控制器生成更长的参考轨迹，以覆盖预测视界
Y_ref_full = generateReferenceCircle(initialState_original(7:12), R_circle, direction_theta, k_steps + 50); % 额外50步用于N_pred
Y_ref_norm = normalize_data([zeros(6, k_steps+50); Y_ref_full], params_state);
Y_ref_norm = Y_ref_norm(7:12, :);
%% MPC控制器参数定义
% --- 权重矩阵 ---
% Q_cost: 输出Y的跟踪误差权重 (n_Output x n_Output)
%         惩罚 (Y - Y_ref)^T * Q_cost * (Y - Y_ref)
Q_cost_diag = [10, 10, 10, 1, 1, 1]; % 对位置误差的权重高于姿态误差
Q_cost = diag(Q_cost_diag); 

% F_cost: 控制输入增量 DeltaU 的权重 (n_InputEigen x n_InputEigen)
%         惩罚 DeltaU^T * F_cost * DeltaU
F_cost_val = 0.1; 
F_cost = eye(n_InputEigen) * F_cost_val;

% R_cost: 控制输入U大小的权重 (n_InputEigen x n_InputEigen)
%         惩罚 U^T * R_cost * U
R_cost_val = 0.01;
R_cost = eye(n_InputEigen) * R_cost_val;

% --- 预测视界 ---
N_pred = 10; % MPC预测步长

% --- 控制输入增量约束 ---
% maxIncremental (标量) 应用于所有输入
max_abs_delta_U = ones(n_InputEigen, 1) * maxIncremental;


%% MPC仿真循环
% --- 初始化仿真变量 ---
% 提升初始状态
X_koopman_current = lift_function(initialState_original_norm, n_StateEigen);

% 存储历史数据
X_koopman_history = zeros(n_StateEigen, k_steps + 1);
Y_history = zeros(n_Output, k_steps + 1);
U_history = zeros(n_InputEigen, k_steps);
Delta_U_history = zeros(n_InputEigen, k_steps); % 存储实际应用的DeltaU

X_koopman_history(:,1) = X_koopman_current;
Y_history(:,1) = C * X_koopman_current;
prev_U = zeros(n_InputEigen, 1); % 上一步的控制输入，初始为0

fprintf('Starting MPC simulation for %d steps...\n', k_steps);
for k = 1:k_steps
    if mod(k, 50) == 0
        fprintf('Simulation step: %d/%d\n', k, k_steps);
    end
    
    % 提取当前视界的参考轨迹
    if k + N_pred -1 <= size(Y_ref_norm, 2)
        Y_ref_horizon = Y_ref_norm(:, k : k + N_pred - 1);
    else % 如果参考轨迹不够长，则重复最后一个值
        num_remaining_ref = size(Y_ref_norm, 2) - k + 1;
        Y_ref_horizon_temp = Y_ref_norm(:, k : end);
        Y_ref_horizon = [Y_ref_horizon_temp, ...
                         repmat(Y_ref_norm(:, end), 1, N_pred - num_remaining_ref)];
    end

    % 调用MPC控制器获取最优控制输入增量序列
    delta_U_optimal_sequence = incrementalMPC(...
                                Q_cost, F_cost, R_cost, N_pred, ...
                                A, B, C, ...
                                X_koopman_current, prev_U, Y_ref_horizon, ...
                                max_abs_delta_U, U_abs_min, U_abs_max, ...
                                n_InputEigen, n_Output, n_StateEigen);
    
    % 应用第一个控制输入增量
    current_delta_U = delta_U_optimal_sequence(:, 1);
    
    % 计算当前控制输入
    current_U = prev_U + current_delta_U;
    
    % (可选) 在此处添加对 current_U 的幅值约束 (如果需要)
    % e.g., current_U = max(min(current_U, U_max), U_min);

    % 更新系统状态 (使用Koopman模型)
    X_koopman_next = A * X_koopman_current + B * current_U;
    
    % 计算系统输出
    Y_next = C * X_koopman_next;
    
    % 存储数据
    Delta_U_history(:, k) = current_delta_U;
    U_history(:, k) = current_U;
    X_koopman_history(:, k+1) = X_koopman_next;
    Y_history(:, k+1) = Y_next;
    
    % 更新状态和前一个输入
    X_koopman_current = X_koopman_next;
    prev_U = current_U;
end
fprintf('MPC simulation finished.\n');

%% 绘制结果
time_vec = 0:k_steps; % 时间向量，对应 Y_history 和 X_koopman_history
time_vec_input = 1:k_steps; % 时间向量，对应 U_history

% 1. 3D轨迹跟踪效果 (假设Y的前三维是x,y,z坐标)
figure;
plot3(Y_history(1,:), Y_history(2,:), Y_history(3,:), 'b-', 'LineWidth', 1.5);
hold on;
plot3(Y_ref_norm(1,1:k_steps+1), Y_ref_norm(2,1:k_steps+1), Y_ref_norm(3,1:k_steps+1), 'r--', 'LineWidth', 1.5);
plot3(Y_history(1,1), Y_history(2,1), Y_history(3,1), 'go', 'MarkerSize', 10, 'MarkerFaceColor', 'g'); % 起点
plot3(Y_ref_norm(1,1), Y_ref_norm(2,1), Y_ref_norm(3,1), 'ro', 'MarkerSize', 10, 'MarkerFaceColor', 'r'); % 参考起点
xlabel('X position');
ylabel('Y position');
zlabel('Z position');
title('3D Trajectory Tracking');
legend('Actual Trajectory (MPC)', 'Reference Trajectory', 'Actual Start', 'Reference Start');
axis equal;
grid on;
view(3); % 3D视角

% 2. Y中每个通道的跟踪效果
figure;
output_labels = {'Y_1 (Pos X)', 'Y_2 (Pos Y)', 'Y_3 (Pos Z)', 'Y_4 (Orient 1)', 'Y_5 (Orient 2)', 'Y_6 (Orient 3)'};
for i = 1:n_Output
    subplot(n_Output/2, 2, i); % 假设 n_Output 是偶数，例如6 -> 3x2 subplot
    plot(time_vec, Y_history(i,:), 'b-', 'LineWidth', 1);
    hold on;
    plot(time_vec, Y_ref_norm(i,1:k_steps+1), 'r--', 'LineWidth', 1);
    xlabel('Time step (k)');
    ylabel(output_labels{i});
    title(['Tracking of Output Channel: ', output_labels{i}]);
    legend('Actual', 'Reference', 'Location', 'best');
    grid on;
end
sgtitle('Output Channel Tracking Performance'); % Super title for all subplots

% 3. 控制输入U
figure;
input_labels = cell(1, n_InputEigen);
for i=1:n_InputEigen
    input_labels{i} = ['U_{', num2str(i), '}'];
end
for i = 1:n_InputEigen
    subplot(ceil(n_InputEigen/2), 2, i); % 调整subplot布局
    plot(time_vec_input, U_history(i,:), 'm-', 'LineWidth', 1);
    xlabel('Time step (k)');
    ylabel(input_labels{i});
    title(['Control Input: ', input_labels{i}]);
    grid on;
end
sgtitle('Control Inputs U');

% 4. 控制输入增量 DeltaU
figure;
delta_input_labels = cell(1, n_InputEigen);
for i=1:n_InputEigen
    delta_input_labels{i} = ['\Delta U_{', num2str(i), '}'];
end
for i = 1:n_InputEigen
    subplot(ceil(n_InputEigen/2), 2, i); % 调整subplot布局
    plot(time_vec_input, Delta_U_history(i,:), 'c-', 'LineWidth', 1);
    hold on;
    plot(time_vec_input, ones(1,k_steps)*maxIncremental, 'k--');
    plot(time_vec_input, -ones(1,k_steps)*maxIncremental, 'k--');
    xlabel('Time step (k)');
    ylabel(delta_input_labels{i});
    title(['Control Input Increment: ', delta_input_labels{i}]);
    legend('Actual \DeltaU', 'Constraint', 'Location', 'best');
    ylim([-maxIncremental*1.2, maxIncremental*1.2]); % 稍微放大y轴范围
    grid on;
end
sgtitle('Control Input Increments \DeltaU');

disp('Plotting complete.');