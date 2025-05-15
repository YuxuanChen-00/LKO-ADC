%% 将主文件夹添加到目录
addpath(genpath('F:\2 软体机器人建模与控制\Bellow-Koopman')); % 修改为您的实际路径
%% 清屏
clear;
close all;
clc;
%% 基本参数
k_steps = 3000; % 总仿真步数
n_states_original = 36; % 原始状态维度 (x_orig)

%% 加载Koopman算子
% --- Koopman和提升函数 ---
lift_function = @lift_polynomial; 
km_path = '../../data/SorotokiPoly/delay3_lift64_relative_without_norm.mat'; 
koopman_parmas = load(km_path);
A = koopman_parmas.A; % Koopman 状态转移矩阵 (n_StateEigen x n_StateEigen)
B = koopman_parmas.B; % Koopman 输入矩阵 (n_StateEigen x n_InputEigen)
% params_control = koopman_parmas.params_control;
% params_state = koopman_parmas.params_state;

n_StateEigen = size(A,1); % Koopman 状态维度 (提升后的维度)
n_InputEigen = size(B,2); % 输入维度 (U的维度)
n_Output = 6;             % 输出维度 (Y的维度)

% 输出矩阵C: 从Koopman状态提取输出Y
if n_StateEigen < n_states_original
    error('n_StateEigen must be >= n_states_original for the given C matrix structure.');
end
C = [zeros(n_Output, 30), eye(n_Output), zeros(n_Output, n_StateEigen - n_states_original)];

%% 生成参考圆轨迹
R_circle = 10; % 圆半径
% 初始原始状态 (12维)
% initialState_original = [25.19, -5.34,-195.82,1,1,1,33.43,-4.93,-290.35,1,1,1]';
initialState_original = [0, 0, 0,1,1,1,0,0,0,1,1,1]';
initialState_original = repmat(initialState_original, 3, 1);

Y_ref_full = zeros(6, k_steps);
for i = 1:k_steps
    Y_ref_full(:,i) = [0,0,0,1,1,1]';
end

%% 开环仿真循环
% --- 初始化仿真变量 ---
% 提升初始状态

X_koopman_current = lift_function(initialState_original, n_StateEigen);

% 存储历史数据
X_koopman_history = zeros(n_StateEigen, k_steps);
Y_history = zeros(n_Output, k_steps);
U_history = zeros(n_InputEigen, k_steps);


fprintf('Starting OpenLoop simulation for %d steps...\n', k_steps);
for k = 1:k_steps
    if mod(k, 50) == 0
        fprintf('Simulation step: %d/%d\n', k, k_steps);
    end
    
    current_U = inv(C*B)*(Y_ref_full(:,k) - C*A*X_koopman_current);
    % current_U = min(max(current_U, 0), 1);

    % 更新系统状态 (使用Koopman模型)
    X_koopman_next = A * X_koopman_current + B * current_U;
    
    % 计算系统输出
    Y_next = C * X_koopman_next;
    
    % 存储数据
    U_history(:, k) = current_U;
    X_koopman_history(:, k) = X_koopman_next;
    Y_history(:, k) = Y_next;
    
    % 更新状态和前一个输入
    X_koopman_current = X_koopman_next;
    prev_U = current_U;
end
fprintf('OpenLoop Control simulation finished.\n');

%% 绘制结果
time_vec = 1:k_steps; % 时间向量，对应 Y_history 和 X_koopman_history
time_vec_input = 1:k_steps; % 时间向量，对应 U_history

% 1. 3D轨迹跟踪效果 (假设Y的前三维是x,y,z坐标)
figure;
plot3(Y_history(1,:), Y_history(2,:), Y_history(3,:), 'b-', 'LineWidth', 1.5);
hold on;
plot3(Y_ref_full(1,1:k_steps), Y_ref_full(2,1:k_steps), Y_ref_full(3,1:k_steps), 'r--', 'LineWidth', 1.5);
plot3(Y_history(1,1), Y_history(2,1), Y_history(3,1), 'go', 'MarkerSize', 10, 'MarkerFaceColor', 'g'); % 起点
plot3(Y_ref_full(1,1), Y_ref_full(2,1), Y_ref_full(3,1), 'ro', 'MarkerSize', 10, 'MarkerFaceColor', 'r'); % 参考起点
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
    plot(time_vec, Y_ref_full(i,1:k_steps), 'r--', 'LineWidth', 1);
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