%% 获取当前文件所在目录
currentDir = fileparts(mfilename('fullpath'));

% 获取上一级目录
parentDir = fileparts(currentDir);

% 只添加上一级目录本身（不包括其子目录）
addpath(parentDir);

net = load('models\LKO_POLY_network\trained_network_20250606_0923.mat');
net = net.net;
%% 神经网络参数
control_var_name = 'input'; 
state_var_name = 'state';  
params = struct();
params.state_size = 6;                % 特征维度
params.delay_step = 5;                   % 节点个数
params.control_size = 6;                % 控制输入维度
params.hidden_size = 12;               % 隐藏层维度
params.PhiDimensions = 24;
params.initialLearnRate = 0.1;         % 初始学习率
params.minLearnRate = 0.00001;                % 最低学习率
params.num_epochs = 1000;                % 训练轮数
params.L1 = 0;                        % 损失权重1
params.L2 = 1;                        % 损失权重2
params.L3 = 0.000001;                       % 损失权重3
params.batchSize = 8172;           % 批处理大小
params.patience = 10;            % 新增参数
params.lrReduceFactor = 0.2; % 新增参数
%% 加载测试数据
% 获取所有.mat文件列表
test_path = '..\..\data\SorotokiData\MotionData4\FilteredDataPos\50secTest';
file_list = dir(fullfile(test_path, '*.mat'));
num_files = length(file_list);

test_data = {1, num_files};

% 处理数据
for file_idx = 1:num_files
    % 加载数据
    file_path = fullfile(test_path, file_list(file_idx).name);
    data = load(file_path);
    % 合并数据
    control_test = data.(control_var_name);
    state_test = data.(state_var_name);

    % 生成时间延迟数据
    [control_timedelay_test, state_timedelay_test, label_timedelay_test] = ...
        generate_timeDelay_data(control_test, state_test, params.delay_step); 
    
    state_hd_test = polynomial_expansion_td(state_timedelay_test, params.PhiDimensions, params.delay_step);
    label_hd_test = polynomial_expansion_td(label_timedelay_test, params.PhiDimensions, params.delay_step);

    current_test_data = struct('control', control_timedelay_test, 'state', state_timedelay_test, ...
        'label', label_timedelay_test, 'state_hd', state_hd_test, 'label_hd', label_hd_test);
        
    test_data{file_idx} = current_test_data;
end

%% 测试
RMSE = zeros(numel(test_data), 1);
for i = 1:numel(test_data)
    control_test = test_data{i}.control;
    state_test = test_data{i}.state;
    label_test = test_data{i}.label;
    [RMSE(i),~,~] = evaluate_lko_poly(net, control_test, state_test, label_test, params.PhiDimensions, params.delay_step);
end
fprintf('RMSE损失 %f \n', mean(RMSE));

% 遍历每个测试文件
for i = 1:numel(test_data)
    % 加载单个测试文件
    
    % 提取当前轨迹数
    
    control_test = test_data{i}.control;
    state_test = test_data{i}.state;
    label_test = test_data{i}.label;
    [RMSE(i), Y_true, Y_pred] = evaluate_lko_poly(net, control_test, state_test, label_test, params.PhiDimensions, params.delay_step);
    
    
    % 绘制当前轨迹的对比图
    fig = figure('Units', 'normalized', 'Position', [0.1 0.1 0.8 0.8]);
    time = 1:size(Y_true, 2);
    
    % 绘制前6个状态量
    for j = 1:6
        subplot(2, 3, j);
        plot(time, Y_true(j,:), 'b-', 'LineWidth', 1.5); 
        hold on;
        plot(time, Y_pred(j,:), 'r--', 'LineWidth', 1.5);
        title(['Dimension ', num2str(j)]);
        xlabel('Time'); 
        ylabel('Value');
        grid on;
        if j == 1
            legend('True', 'Predicted', 'Location', 'northoutside');
        end
    end
end