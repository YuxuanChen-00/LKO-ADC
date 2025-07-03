% 获取当前文件所在目录
currentDir = fileparts(mfilename('fullpath'));

% 获取上一级目录
parentDir = fileparts(currentDir);

% 只添加上一级目录本身（不包括其子目录）
addpath(parentDir);
%% 清屏
clear;
close all;
clc;
%% 基本参数
k_steps = 1000; % 总仿真步数
n_states_original = 6; % 原始状态维度 (x_orig)
control_gamma = 0.000;
gamma1 = 0.0000000000*ones(6, 6);
gamma2 = 0.0000000000;

%% 加载Koopman算子
% --- Koopman和提升函数 ---
delay_time = 7;
target_dimensions = 24;
lift_function = @polynomial_expansion_td; 
km_path = '../koopman_model/poly_delay7_lift24.mat'; % 修改为您的实际路径
koopman_parmas = load(km_path);

% 假设当前Koopman模型是真实模型
A = koopman_parmas.A; % Koopman 状态转移矩阵 (n_StateEigen x n_StateEigen)
B = koopman_parmas.B; % Koopman 输入矩阵 (n_StateEigen x n_InputEigen)


n_StateEigen = size(A,1); % Koopman 状态维度 (提升后的维度)
n_InputEigen = size(B,2); % 输入维度 (U的维度)
n_Output = 6;             % 输出维度 (Y的维度)

C = [eye(n_Output,n_Output), zeros(n_Output, n_StateEigen - n_Output)];

% 假设参考模型的初始值如下
A_1 = C*((A+0.01*A)-eye(n_StateEigen));
B_1 = C*(B+0.01*B);
K_1 = [A_1, B_1];

Ac = A_1;
Bc = B_1;
Kc = [Ac, Bc];
Kc_init = Kc;

%% --- 控制输入约束 ---
maxIncremental = 0.1; % 最大控制输入增量 (标量，假设对所有输入相同)
U_abs_min = [0;0;0;0;0;0];
U_abs_max = [5;5;5;5;5;5];


%% LQR控制器
% 定义LQR权重矩阵
Q = diag([10, 10, 10, 10,10, 10]); % 状态误差惩罚矩阵，对两个状态都给予较高的惩罚
R = diag([1,1,1,1,1,1]);      % 控制输入惩罚矩阵，对控制量给予适中惩罚

% 使用 dlqr 函数计算LQR反馈增益矩阵 K
[K, ~, ~] = dlqr(A, B, Q, R);

%% 生成参考圆轨迹
% 初始原始状态 (12维)
initialState_original = [14.23;-4.42;-215.35;18.95;-22.45;-339.89];
initialState_original = repmat(initialState_original, delay_time, 1);
R_circle2 = 45; % 圆半径
R_circle1 = 6; % 圆半径
weights = linspace(0,1,1000);

% 为MPC控制器生成更长的参考轨迹，以覆盖预测视界
center1 = initialState_original(1:3);
Y_ref1 = generateReferenceCircle(center1, R_circle1, k_steps); % 额外50步用于N_pred
to_center1 = center1*(1-weights)+Y_ref1(:,1)*weights;
Y_ref1 = [to_center1, Y_ref1, repmat(Y_ref1(:, end), 1, 20)];

center2 = initialState_original(4:6);
Y_ref2 = generateReferenceCircle(center2, R_circle2, k_steps); % 额外50步用于N_pred
to_center2 = center2*(1-weights)+Y_ref2(:,1)*weights;
Y_ref2 = [to_center2, Y_ref2, repmat(Y_ref2(:, end), 1, 20)];

Y_ref = [Y_ref1;Y_ref2];
% Y_ref = [repmat(center1, 1, 1000); repmat(center2, 1, 1000)];

current_ref_delay = repmat(Y_ref(:, 1), delay_time, 1);
k_steps = k_steps + 1000;
d_Y_ref = [zeros(6,1), diff(Y_ref, 1, 2)];

%% 仿真循环
% --- 初始化仿真变量 ---
% 提升初始状态
X_koopman_current = lift_function(initialState_original, target_dimensions, delay_time);

% 存储历史数据
X_koopman_history = zeros(n_StateEigen, k_steps + 1);
Y_history = zeros(n_Output, k_steps + 1);
U_history = zeros(n_InputEigen, k_steps);
phi_error_slide_window = prediction_history(n_states_original, 10);
phi_slide_window = prediction_history(n_StateEigen + n_InputEigen, 10);
last_control_input = [2;2;2;2;2;2];

X_koopman_history(:,1) = X_koopman_current;
Y_history(:,1) = C * X_koopman_current;

fprintf('Starting simulation for %d steps...\n', k_steps);
for k = 1:k_steps
    if mod(k, 50) == 0
        fprintf('Simulation step: %d/%d\n', k, k_steps);
    end

    % 计算当前误差
    y_ref = d_Y_ref(:, k+1);
    x_koopman_ref = lift_function(current_ref_delay, target_dimensions, delay_time);
    x_current = X_koopman_current(1:6);

    e = y_ref -  x_current;

    % 计算当前控制输入
    current_U = get_adaptive_control_input(e, y_ref, x_koopman_ref, last_control_input, ...
        Ac, Bc, control_gamma, U_abs_max, U_abs_min, maxIncremental);

    % 更新系统状态 (使用Koopman模型)
    X_koopman_next = A * X_koopman_current + B * current_U;
    X_pred = Ac * X_koopman_current + Bc * current_U + x_current;
    delta_x = X_koopman_next(1:6) - X_pred;

    % 计算系统输出
    Y_next = C * X_koopman_next;

    % 存储数据
    U_history(:, k) = current_U;
    X_koopman_history(:, k+1) = X_koopman_next;
    Y_history(:, k+1) = Y_next;
    last_control_input = current_U;
    current_ref_delay = [Y_ref(:, k+1); current_ref_delay(1:end-n_Output)];

    % 更新状态和前一个输入
    X_koopman_current = X_koopman_next;

    % 更新Koopman算子
    phi_slide_window = phi_slide_window.getdata([X_koopman_next;current_U]);
    phi_error_slide_window = phi_error_slide_window.getdata(delta_x);
    delta_K = update_K(e, [X_koopman_next;current_U], phi_error_slide_window, phi_slide_window, gamma1, gamma2);
    Kc = Kc + delta_K;
    Ac = Kc(:, 1:n_StateEigen);
    Bc = Kc(:, n_StateEigen+1:end);
end
fprintf('MPC simulation finished.\n');


%% 计算均方根误差
mse1 = calculateRMSE(Y_ref(1:3, 1:k_steps+1), Y_history(1:3,:));
mse2 = calculateRMSE(Y_ref(4:6, 1:k_steps+1), Y_history(4:6,:));
fprintf('第一关节轨迹跟踪均方根误差为: %2f, 第二关节轨迹跟踪的均方根误差为: %2f\n', mse1, mse2);


% % 绘制结果
time_vec = 0:k_steps; % 时间向量，对应 Y_history 和 X_koopman_history
time_vec_input = 1:k_steps; % 时间向量，对应 U_history

% 1. 3D轨迹跟踪效果 (假设Y的前三维是x,y,z坐标)
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
legend('Actual Trajectory (MPC)', 'Reference Trajectory', 'Actual Start', 'Reference Start');
axis equal;
grid on;

% 2. 3D轨迹跟踪效果 (假设Y的前三维是x,y,z坐标)
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
view(3); % 3D视角


% 2. Y中每个通道的跟踪效果
figure;
output_labels = {'Y_1 (Pos X)', 'Y_1 (Pos Y)', 'Y_1 (Pos Z)','Y_2 (Pos X)', 'Y_2 (Pos Y)', 'Y_2 (Pos Z)'};
for i = 1:n_Output
    subplot(n_Output, 1, i); % 假设 n_Output 是偶数，例如6 -> 3x2 subplot
    plot(time_vec, Y_history(i,:), 'b-', 'LineWidth', 1);
    hold on;
    plot(time_vec, Y_ref(i,1:k_steps+1), 'r--', 'LineWidth', 1);
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


disp('Plotting complete.');