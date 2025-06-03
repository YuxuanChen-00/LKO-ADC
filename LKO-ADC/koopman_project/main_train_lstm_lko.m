mainFolder = fileparts(mfilename('fullpath'));
% 添加主文件夹及其所有子文件夹到路径
addpath(genpath(mainFolder));
%% 参数设置
% 生成数据参数
control_var_name = 'input'; 
state_var_name = 'state';    
loss_pred_step = 1;
% 神经网络参数
params = struct();
params.state_size = 6;                % 特征维度
params.delay_step = 7;                   % 节点个数
params.control_size = 6;                % 控制输入维度
params.hidden_size = 8;               % 隐藏层维度
params.PhiDimensions = 24;              % 高维特征维度
params.output_size = params.PhiDimensions - params.state_size;
params.initialLearnRate = 0.01;         % 初始学习率
params.minLearnRate = 0;                % 最低学习率
params.num_epochs = 1000;                % 训练轮数
params.L1 = 1;                        % 损失权重1
params.L2 = 1;                        % 损失权重2
params.L3 = 0.0001;                       % 损失权重3
params.batchSize = 128;           % 批处理大小
params.patience = 20;            % 新增参数
params.lrReduceFactor = 0.2; % 新增参数

train_path = 'data\SorotokiData\MotionData4\FilteredDataPos\40minTrain';
test_path = 'data\SorotokiData\MotionData4\FilteredDataPos\50secTest';
model_save_path = 'models\LKO_LSTM_SorotokiPositionData_network1\';

if ~exist(model_save_path, 'dir')
    % 如果不存在则创建文件夹
    mkdir(model_save_path);
    disp(['文件夹 "', model_save_path, '" 已创建']);
end

%% 加载训练数据
% 获取所有.mat文件列表
file_list = dir(fullfile(train_path, '*.mat'));
num_files = length(file_list);

control_train = [];
label_train = [];
state_train = [];

% 处理数据
for file_idx = 1:num_files
    % 加载数据
    file_path = fullfile(train_path, file_list(file_idx).name);
    data = load(file_path);
    % 合并数据

    % 生成时间延迟数据
    [control_timedelay_train, state_timedelay_train, label_timedelay_train] = ...
        generate_lstm_data(data.(control_var_name), data.(state_var_name), params.delay_step, loss_pred_step); 

    control_train = cat(2, control_train, control_timedelay_train);
    state_train = cat(2, state_train, state_timedelay_train);
    label_train = cat(2, label_train, label_timedelay_train);
end

train_data.control_sequences = control_timedelay_train;
train_data.state_sequences = state_timedelay_train;
train_data.label_sequences = label_timedelay_train;

% 归一化数据
% [norm_control_train, params_control] = normalize_data(control_train);
% [norm_state_train, params_state] = normalize_data(state_train);
% save([model_save_path, 'norm_params'], 'params_state', 'params_control')


%% 加载测试数据
% 获取所有.mat文件列表
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
        generate_lstm_data(control_test, state_test, params.delay_step, loss_pred_step); 
    
    current_test_data = struct('control', control_timedelay_test, 'state', state_timedelay_test, ...
        'label', label_timedelay_test);
        
    test_data{file_idx} = current_test_data;

end

% 归一化数据
% [norm_control_test, params_control] = normalize_data(control_test, params_control);
% [norm_state_test, params_state] = normalize_data(state_test, params_state);


%% 训练
[net, A, B] = train_lstm_lko(params, train_data, test_data);
save([model_save_path, 'trained_network.mat'], 'net', 'A', 'B');  % 保存整个网络


%% 测试
RMSE = zeros(numel(test_data), 1);
for i = 1:numel(test_data)
    control_test = test_data{i}.control;
    state_test = test_data{i}.state;
    label_test = test_data{i}.label;
    [RMSE(i),~,~] = evaluate_lstm_lko(net, control_test, state_test, label_test, params.delay_step);
end
fprintf('RMSE损失 %f \n', mean(RMSE));



% 遍历每个测试文件
for i = 1:numel(test_data)
    % 加载单个测试文件
    
    % 提取当前轨迹数
    
    control_test = test_data{i}.control;
    state_test = test_data{i}.state;
    label_test = test_data{i}.label;
    [RMSE(i), Y_true, Y_pred] = evaluate_lstm_lko2(net, control_test, state_test, label_test, params.delay_step);
    
    
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


