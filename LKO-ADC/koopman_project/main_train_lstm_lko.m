mainFolder = fileparts(mfilename('fullpath'));
% 添加主文件夹及其所有子文件夹到路径
addpath(genpath(mainFolder));
%% 参数设置
% 生成数据参数
time_step = 2;
loss_pred_step = 10;
control_var_name = 'input'; 
state_var_name = 'state';    
% 神经网络参数
params = struct();
params.state_size = 6;                % 特征维度
params.time_step = time_step;                   % 节点个数
params.control_size = 6;                % 控制输入维度
params.hidden_size = 64;               % 隐藏层维度
params.PhiDimensions = 68;              % 高维特征维度
params.output_size = params.PhiDimensions - params.state_size*params.time_step;
params.initialLearnRate = 1e-2;         % 初始学习率
params.minLearnRate = 0;                % 最低学习率
params.num_epochs = 10000;                % 训练轮数
params.L1 = 100;                        % 损失权重1
params.L2 = 1;                        % 损失权重2
params.L3 = 0;                       % 损失权重3
params.batchSize = 1024;           % 批处理大小
params.patience = 20;            % 新增参数
params.lrReduceFactor = 0.2; % 新增参数

train_path = 'data\SorotokiData\MotionData2_without_Direction\trainData';
test_path = 'data\SorotokiData\\MotionData2_without_Direction\testData';
model_save_path = ['models\LKO_LSTM_directiondata_delay' num2str(time_step) 'pred' num2str(loss_pred_step) ...
    'H' num2str(params.hidden_size) 'P' num2str(params.PhiDimensions) '_network\'];

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
        generate_lstm_data(data.(control_var_name), data.(state_var_name), time_step, loss_pred_step); 

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
        generate_lstm_data(control_test, state_test, time_step, loss_pred_step); 
    
    current_test_data = struct('control', control_timedelay_test, 'state', state_timedelay_test, ...
        'label', label_timedelay_test);
        
    test_data{file_idx} = current_test_data;

end

% 归一化数据
% [norm_control_test, params_control] = normalize_data(control_test, params_control);
% [norm_state_test, params_state] = normalize_data(state_test, params_state);


%% 训练

[last_model, A, B] = train_lstm_lko(params, train_data, test_data, model_save_path);


