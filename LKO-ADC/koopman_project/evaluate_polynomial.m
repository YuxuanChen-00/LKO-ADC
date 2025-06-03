%% 主程序框架
mainFolder = fileparts(mfilename('fullpath'));
addpath(genpath(mainFolder));

%% 参数设置
delay_time = 3;
target_dimensions = 12;
lift_function = @polynomial_expansion_td;
train_path = 'data\SorotokiData\MotionData4\FilteredDataPos\40minTrain';
test_path = 'data\SorotokiData\MotionData4\FilteredDataPos\50secTest';
km_save_path = 'models\SorotokiPoly\delay3_lift64_relative.mat'; 
control_var_name = 'input'; 
state_var_name = 'state';    
state_window = 1:6;
predict_window = 1:579;

%% 训练阶段
% 加载训练数据
file_list = dir(fullfile(train_path, '*.mat'));
num_train_files = length(file_list);

control_sequences = [];
state_sequences = [];

for file_idx = 1:num_train_files
    file_path = fullfile(train_path, file_list(file_idx).name);
    data = load(file_path);
    control_sequences = cat(2, control_sequences, data.(control_var_name));
    state_sequences = cat(2, state_sequences, data.(state_var_name));
end

% 生成时间延迟数据
[control_timedelay, state_timedelay, label_timedelay] = ...
    generate_timeDelay_data(control_sequences, state_sequences, delay_time); 

% 计算Koopman算子
state_timedelay_phi = lift_function(state_timedelay, target_dimensions, delay_time);
label_timedelay_phi = lift_function(label_timedelay, target_dimensions, delay_time);
[A, B] = koopman_operator(control_timedelay, state_timedelay_phi, label_timedelay_phi);
save(km_save_path, "A", "B")

%% 测试阶段
% 加载测试数据
test_files = dir(fullfile(test_path, '*.mat'));
num_test_files = length(test_files);

% 预分配结果存储
all_RMSE = zeros(num_test_files, 1);
all_predictions = cell(num_test_files, 1);
all_groundtruth = cell(num_test_files, 1);

% 创建结果保存目录
if ~exist('results', 'dir')
    mkdir('results')
end

% 遍历每个测试文件
for test_idx = 1:num_test_files
    % 加载单个测试文件
    test_file = fullfile(test_path, test_files(test_idx).name);
    test_data = load(test_file);
    
    % 提取当前轨迹数据
    current_control = test_data.(control_var_name);
    current_state = test_data.(state_var_name);
    
    % 生成时间延迟数据（单个轨迹内处理）
    [control_td, state_td, label_td] = ...
        generate_timeDelay_data(current_control, current_state, delay_time);
    
    % 提升维度
    state_td_phi = lift_function(state_td, target_dimensions, delay_time);
    
    % 执行多步预测
    Y_true = label_td(state_window, predict_window+ 30 - delay_time);
    Y_pred = predict_multistep(A, B, control_td(:, predict_window + 30 - delay_time),...
        state_td_phi(:, predict_window(1)+ 30 - delay_time),...
        predict_window(end)-predict_window(1)+1);
    Y_pred = Y_pred(state_window, :);
    
    % 存储结果
    all_RMSE(test_idx) = calculateRMSE(Y_pred, Y_true);
    all_predictions{test_idx} = Y_pred;
    all_groundtruth{test_idx} = Y_true;
    
    % 绘制当前轨迹的对比图
    fig = figure('Units', 'normalized', 'Position', [0.1 0.1 0.8 0.8]);
    time = 1:size(Y_true, 2);
    
    % 绘制前6个状态量
    for i = 1:6
        subplot(2, 3, i);
        plot(time, Y_true(i,:), 'b-', 'LineWidth', 1.5); 
        hold on;
        plot(time, Y_pred(i,:), 'r--', 'LineWidth', 1.5);
        title(['Dimension ', num2str(i)]);
        xlabel('Time'); 
        ylabel('Value');
        grid on;
        if i == 1
            legend('True', 'Predicted', 'Location', 'northoutside');
        end
    end
    % 保存图像
    % set(fig, 'Color', 'w');
    % sgtitle(['Test Case ', num2str(test_idx), ' Prediction Results']);
    % saveas(fig, fullfile('results', ['test_case_', num2str(test_idx), '.png']));
    % close(fig);
end

%% 显示统计结果
% 计算全局统计量
mean_RMSE = mean(all_RMSE);
std_RMSE = std(all_RMSE);
min_RMSE = min(all_RMSE);
max_RMSE = max(all_RMSE);

% 显示结果
disp('================= 综合测试结果 =================');
disp(['平均 RMSE: ', num2str(mean_RMSE)]);
disp(['标准差: ', num2str(std_RMSE)]);
disp(['最小值: ', num2str(min_RMSE)]);
disp(['最大值: ', num2str(max_RMSE)]);
disp('各测试案例RMSE:');
disp(all_RMSE');

%% 特征值分析（保持原代码）
[V, D] = eig(A);
eigenvalues = diag(D);

fig = figure;
theta = linspace(0, 2*pi, 100);
plot(cos(theta), sin(theta), 'k--', 'LineWidth', 1.5);
hold on;
scatter(real(eigenvalues), imag(eigenvalues), 'ro', 'filled');
axis equal; grid on;
xlabel('Real'); ylabel('Imaginary');
title('Eigenvalue Distribution on Unit Circle');
legend('Unit Circle', 'Eigenvalues');
% saveas(fig, fullfile('results', 'eigenvalue_distribution.png'));