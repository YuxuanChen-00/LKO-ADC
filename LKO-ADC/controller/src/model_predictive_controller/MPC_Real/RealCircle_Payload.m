%% 清屏
clear;
close all;
clc;

% 获取当前文件所在目录
currentDir = fileparts(mfilename('fullpath'));

% 获取上一级目录
parentDir = fileparts(currentDir);

% 只添加上一级目录本身（不包括其子目录）
addpath(parentDir);

%% 载入数据采集卡
delete (instrfindall);
serialforce = serial('COM2', 'BaudRate', 115200, 'Parity', 'none',...
                'DataBits', 8, 'StopBits', 1);
deviceDescription = 'PCI-1716,BID#0'; % Analog input card
deviceDescription_1 = 'PCI-1727U,BID#0'; % Analog input card  
AOchannelStart = int32(1);
AOchannelCount = int32(7);  
BDaq = NET.addAssembly('Automation.BDaq4');
errorCode = Automation.BDaq.ErrorCode.Success;
instantAoCtrl = Automation.BDaq.InstantAoCtrl();
instantAoCtrl.SelectedDevice = Automation.BDaq.DeviceInformation(...
    deviceDescription_1);
scaleData = NET.createArray('System.Double', int32(64));

global onemotion_data;
serialvicon = serial('COM1');
set(serialvicon,'BaudRate',115200);
set(serialvicon,'BytesAvailableFcnMode','Terminator'); 
set(serialvicon,'Terminator','LF');
set(serialvicon,'BytesAvailableFcn',{@ReceiveVicon});
fopen(serialvicon);
%% 基本参数
k_steps = 600; % 总控制步数
num_tracker = 3;
num_state = (num_tracker-1)*6;
lastAoData = [0,0,0,0,0,0,0];
weight = [1,2,3,4,5]';
position_index = [1,2,3,7,8,9];

% 采样频率是控制频率的五倍，每四个点做一次平均作为当前状态
control_freq = 10;  % 控制频率10Hz
sampling_freq = 100;  % 采样频率100Hz
controlRate = robotics.Rate(control_freq);  % 控制更新速率
samplingRate = robotics.Rate(sampling_freq);  % 采样更新速率
num_samples = 5;  % 每3个采样点做平均

% 获得初始旋转矩阵和初始点位置
prev_U = [2.5,2.5,2.5,1.5,1.5,1.5]; % 上一步的控制输入，初始为2
AoData = [prev_U,3];
prev_U = prev_U';
linearPressureControl(AoData, lastAoData, samplingRate, instantAoCtrl,...
    scaleData,AOchannelStart, AOchannelCount)
initRate = robotics.Rate(0.2);
waitfor(initRate);
[initRotationMatrix, initPosition] = getInitState(onemotion_data);
last_sample = transferVicon2Base(onemotion_data, initRotationMatrix, initPosition);


%% 加载Koopman算子
% --- Koopman和提升函数 ---
delay_time = 8;
target_dimensions = 30;
lift_function = @polynomial_expansion_td; 
trajectory_function = @generateReferenceCircle;
km_path = '../koopman_model/update_poly_delay8_lift30.mat'; % 修改为您的实际路径
koopman_parmas = load(km_path);
A = koopman_parmas.A; % Koopman 状态转移矩阵 (n_StateEigen x n_StateEigen)
B = koopman_parmas.B; % Koopman 输入矩阵 (n_StateEigen x n_InputEigen)

n_StateEigen = size(A,1); % Koopman 状态维度 (提升后的维度)
n_InputEigen = size(B,2); % 输入维度 (U的维度)
n_Output = 6;             % 输出维度 (Y的维度)

C = [eye(n_Output,n_Output), zeros(n_Output, n_StateEigen - n_Output)];
%% --- 控制输入约束 ---
maxIncremental = 0.05; % 最大控制输入增量 (标量，假设对所有输入相同)
U_abs_min = [0;0;0;0;0;0];
U_abs_max = [6;6;6;6;6;6];

%% 生成参考圆轨迹
R_circle2 = 50; % 圆半径
R_circle1 = 10; % 圆半径
% 初始原始状态 (12维)
initialState_original = [initPosition.P1; initPosition.P2];
initialState_original = repmat(initialState_original, delay_time, 1);


weights = linspace(0,1,100);

% 为MPC控制器生成更长的参考轨迹，以覆盖预测视界
center1 = initialState_original(1:3);
Y_ref1 = generateReferenceCircle(center1, R_circle1, k_steps); % 额外50步用于N_pred
Y_ref1 = [repmat(Y_ref1(:, 1), 1, 100), Y_ref1, repmat(Y_ref1(:, end), 1, 20)];

center2 = initialState_original(4:6);
Y_ref2 = generateReferenceCircle(center2, R_circle2, k_steps); % 额外50步用于N_pred
Y_ref2 = [repmat(Y_ref2(:, 1), 1, 100), Y_ref2, repmat(Y_ref2(:, end), 1, 20)];

Y_ref = [Y_ref1;Y_ref2];

k_steps = k_steps + 100;

%% MPC控制器参数定义
% --- 权重矩阵 ---
% Q_cost: 输出Y的跟踪误差权重 (n_Output x n_Output)
%         惩罚 (Y - Y_ref)^T * Q_cost * (Y - Y_ref)
Q_cost_diag = [10, 10, 10, 10, 10, 10]; 
Q_cost = diag(Q_cost_diag); 

% F_cost: 控制输入增量 DeltaU 的权重 (n_InputEigen x n_InputEigen)
%         惩罚 DeltaU^T * F_cost * DeltaU
F_cost_val = 0.01; 
F_cost = eye(n_InputEigen) * F_cost_val;

% R_cost: 控制输入U大小的权重 (n_InputEigen x n_InputEigen)
%         惩罚 U^T * R_cost * U
R_cost_val = 0.01;
R_cost = eye(n_InputEigen) * R_cost_val;

% --- 预测视界 ---
N_pred = 4; % MPC预测步长

% --- 控制输入增量约束 ---
% maxIncremental (标量) 应用于所有输入
max_abs_delta_U = ones(n_InputEigen, 1) * maxIncremental;


%% MPC仿真循环
% --- 初始化仿真变量 ---
% 提升初始状态
X_koopman_current = lift_function(initialState_original, target_dimensions, delay_time);

% 存储历史数据
X_koopman_history = zeros(n_StateEigen, k_steps + 1);
Y_history = zeros(n_Output, k_steps + 1);
U_history = zeros(n_InputEigen, k_steps);
Delta_U_history = zeros(n_InputEigen, k_steps); % 存储实际应用的DeltaU
Obs_history = zeros(num_state, k_steps + 1);

X_koopman_history(:,1) = X_koopman_current;
Y_history(:,1) = C * X_koopman_current;
Obs_history(:, 1) = last_sample;


% 状态时间序列
state_sequence = initialState_original;

% 控制循环
tic;
for k = 1:k_steps
    
    % 提取当前视界的参考轨迹
    if k + N_pred -1 <= size(Y_ref, 2)
        Y_ref_horizon = Y_ref(:, k : k + N_pred - 1);
    else % 如果参考轨迹不够长，则重复最后一个值
        num_remaining_ref = size(Y_ref, 2) - k + 1;
        Y_ref_horizon_temp = Y_ref(:, k : end);
        Y_ref_horizon = [Y_ref_horizon_temp, ...
                         repmat(Y_ref(:, end), 1, N_pred - num_remaining_ref)];
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
    
    current_U = max(min(current_U, U_abs_max), 0);

    % 控制输入到系统中
    AoData = [current_U', 3];
    max_input = [6,6,6,6,6,6,3];
    AoData = min(AoData, max_input);
    linearPressureControl(AoData, lastAoData, samplingRate, instantAoCtrl,...
        scaleData,AOchannelStart, AOchannelCount)
    lastAoData = AoData;
    
    % 控制频率更新
    waitfor(controlRate);
    time_elapsed = toc;
    disp(['第' num2str(k) '次循环, 用时' num2str(time_elapsed) '秒']);
    tic;

    % 获取机器人位姿
   [current_state, current_raw, last_sample] = sampleAndFilterViconData(samplingRate, ...
       num_samples, initRotationMatrix, initPosition, weight, last_sample);
   
   if any(isinf(current_state), 'all')
        X_koopman_next = A * X_koopman_current + B * current_U;
   else
       state_sequence = [current_state(position_index);state_sequence(1:end-6)];
       X_koopman_next = lift_function(state_sequence, target_dimensions, delay_time);
   end
   
    
    % 存储数据
    Delta_U_history(:, k) = current_delta_U;
    U_history(:, k) = current_U;
    X_koopman_history(:, k+1) = X_koopman_next;
    Y_history(:, k+1) = X_koopman_next(1:6);
    Obs_history(:, k+1) = current_state;
    
    % 更新状态和前一个输入
    X_koopman_current = X_koopman_next;
    prev_U = current_U;
end
fprintf('MPC simulation finished.\n');


%% 计算均方根误差
mse1 = calculateRMSE(Y_ref(1:3, 101:k_steps+1), Y_history(1:3,101:end));
mse2 = calculateRMSE(Y_ref(4:6, 101:k_steps+1), Y_history(4:6,101:end));
fprintf('第一关节轨迹跟踪均方根误差为: %2f, 第二关节轨迹跟踪的均方根误差为: %2f\n', mse1, mse2);

save('60secCircleTrack_payload_10g.mat', 'Y_history', 'U_history', 'Y_ref');


%% 绘制结果
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
% legend('Actual Trajectory (MPC)', 'Reference Trajectory', 'Actual Start', 'Reference Start');
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