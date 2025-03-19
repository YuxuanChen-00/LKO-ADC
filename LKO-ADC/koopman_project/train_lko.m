mainFolder = fileparts(mfilename('fullpath'));
% 添加主文件夹及其所有子文件夹到路径
addpath(genpath(mainFolder));
%% 参数设置
time_step = 3;
target_dimensions = 64;
lift_function = @polynomial_expansion;
train_path = 'data\BellowData\rawData\trainData';
test_path = 'data\BellowData\rawData\testData';
model_save_path = 'models\LKO_GCN_3step_network';
control_var_name = 'U_list'; 
state_var_name = 'X_list';    
state_window = 25:36;
predict_step = 1000;

%% 加载训练数据
% 获取所有.mat文件列表
file_list = dir(fullfile(train_path, '*.mat'));
num_files = length(file_list);

% 初始化三维存储数组
control_sequences = [];  % c x N
state_sequences = [];    % dm x N

% 处理数据
for file_idx = 1:num_files
    % 加载数据
    file_path = fullfile(train_path, file_list(file_idx).name);
    data = load(file_path);
    % 合并数据
    control_sequences = cat(2, control_sequences, data.(control_var_name));
    state_sequences = cat(2, state_sequences, data.(state_var_name));
end

% 归一化数据
[norm_control, params_control] = normalize_data(control_sequences);
[norm_state, params_state] = normalize_data(state_sequences);


% 生成时间延迟数据
[control_timedelay, state_timedelay, label_timedelay] = ...
    generate_gcn_data(norm_control, norm_state, time_step); 
%% 训练
train_data.control_sequences = control_timedelay;
train_data.state_sequences = state_timedelay;
train_data.label_sequences = label_timedelay;

[last_model, A, B] = train_gcn_lko(train_data, model_save_path);



%% 预测
state_timedelay_phi = lift_function(state_timedelay, target_dimensions);
Y_true = label_timedelay(state_window, 1:predict_step);
Y_pred = predict_multistep(A, B, control_timedelay, state_timedelay_phi(:,1), predict_step);
Y_pred = Y_pred(state_window, 1:predict_step);


%% 绘图
% Y_true 是 12×t 的真实值矩阵
% Y_pred 是 12×t 的预测值矩阵

figure('Units', 'normalized', 'Position', [0.1 0.1 0.8 0.8]); % 全屏大窗口
time = 1:size(Y_true, 2); % 生成时间轴

% 绘制12个子图（3行×4列）
for i = 1:12
    subplot(3, 4, i);

    % 绘制真实值和预测值曲线
    plot(time, Y_true(i,:), 'b-', 'LineWidth', 1.5); hold on;
    plot(time, Y_pred(i,:), 'r--', 'LineWidth', 1.5);

    % 美化图形
    title(['Dimension ', num2str(i)]);
    xlabel('Time'); 
    ylabel('Value');
    grid on;

    % 只在第一个子图显示图例
    if i == 1
        legend('True', 'Predicted', 'Location', 'northoutside');
    end

    % 统一坐标轴范围（可选）
    % ylim([min(Y_true(:)), max(Y_true(:))]);
end

% 调整子图间距
set(gcf, 'Color', 'w'); % 设置背景为白色
ha = findobj(gcf, 'type', 'axes');
set(ha, 'FontSize', 9); % 统一字体大小
sgtitle('True vs Predicted Values across 12 Dimensions'); % 总标题

% 保存图像（可选）
% saveas(gcf, 'Fitting_Comparison.png');