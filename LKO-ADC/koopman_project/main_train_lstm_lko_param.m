mainFolder = fileparts(mfilename('fullpath'));
addpath(genpath(mainFolder));

%% 参数迭代设置
param_combinations = struct();
param_combinations.delay_time = 7;     % 迭代的延迟步长值
param_combinations.hidden_size = 8:2:20;   % 迭代的隐藏层维度值
param_combinations.phi_dimensions = 24; % 迭代的Phi维度值

% 生成所有参数组合
delay_list = param_combinations.delay_time;
hidden_list = param_combinations.hidden_size;
phi_list = param_combinations.phi_dimensions;
[delay_grid, hidden_grid, phi_grid] = ndgrid(delay_list, hidden_list, phi_list);
param_sets = [delay_grid(:), hidden_grid(:), phi_grid(:)];

%% 固定参数配置
base_params = struct();
base_params.state_size = 6;            % 固定：特征维度
base_params.control_size = 6;          % 固定：控制输入维度
base_params.initialLearnRate = 0.01;   % 固定：初始学习率
base_params.minLearnRate = 0;          % 固定：最低学习率
base_params.num_epochs = 2000;         % 固定：训练轮数
base_params.L1 = 0;                    % 固定：损失权重1
base_params.L2 = 1;                    % 固定：损失权重2
base_params.L3 = 0.0001;               % 固定：损失权重3
base_params.batchSize = 128;           % 固定：批处理大小
base_params.patience = 20;             % 固定：早停耐心值
base_params.lrReduceFactor = 0.2;      % 固定：学习率衰减因子

%% 路径设置
train_path = 'data\SorotokiData\MotionData4\FilteredDataPos\40minTrain';
test_path = 'data\SorotokiData\MotionData4\FilteredDataPos\50secTest';
main_model_save_path = 'models\LKO_LSTM_SorotokiPositionData_Iterations\';

if ~exist(main_model_save_path, 'dir')
    mkdir(main_model_save_path);
    disp(['主文件夹 "', main_model_save_path, '" 已创建']);
end

%% 主循环：遍历所有参数组合
for i_set = 1:size(param_sets, 1)
    % 获取当前参数组合
    curr_delay = param_sets(i_set, 1);
    curr_hidden = param_sets(i_set, 2);
    curr_phi = param_sets(i_set, 3);
    
    % 创建参数结构体
    params = base_params;
    params.delay_step = curr_delay;
    params.hidden_size = curr_hidden;
    params.PhiDimensions = curr_phi;
    params.output_size = curr_phi - params.state_size; % 计算输出维度
    
    % 创建模型保存路径（包含参数值）
    model_save_path = sprintf('%sdelay%d_hid%d_phi%d\\', ...
                             main_model_save_path, ...
                             curr_delay, ...
                             curr_hidden, ...
                             curr_phi);
                         
    if ~exist(model_save_path, 'dir')
        mkdir(model_save_path);
        disp(['参数文件夹 "', model_save_path, '" 已创建']);
    end
    
    % 显示当前参数组合
    fprintf('\n==== 训练参数组合 [%d/%d]: delay=%d, hidden=%d, phi=%d ====\n', ...
            i_set, size(param_sets, 1), curr_delay, curr_hidden, curr_phi);
    
    %% 加载并生成训练数据（使用当前delay_step）
    file_list = dir(fullfile(train_path, '*.mat'));
    num_files = length(file_list);

    control_train = [];
    state_train = [];
    label_train = [];

    for file_idx = 1:num_files
        file_path = fullfile(train_path, file_list(file_idx).name);
        data = load(file_path);
        
        [control_timedelay, state_timedelay, label_timedelay] = ...
            generate_lstm_data(data.input, data.state, params.delay_step, 1); 
        
        control_train = cat(2, control_train, control_timedelay);
        state_train = cat(2, state_train, state_timedelay);
        label_train = cat(2, label_train, label_timedelay);
    end

    train_data.control_sequences = control_train;
    train_data.state_sequences = state_train;
    train_data.label_sequences = label_train;
    
    %% 加载测试数据
    test_data = {};
    test_files = dir(fullfile(test_path, '*.mat'));
    
    for file_idx = 1:length(test_files)
        file_path = fullfile(test_path, test_files(file_idx).name);
        data = load(file_path);
        
        [control_timedelay, state_timedelay, label_timedelay] = ...
            generate_lstm_data(data.input, data.state, params.delay_step, 1);
        
        current_test = struct(...
            'control', control_timedelay, ...
            'state', state_timedelay, ...
            'label', label_timedelay);
        
        test_data{end+1} = current_test;
    end

    %% 训练模型
    [net, A, B] = train_lstm_lko(params, train_data, test_data);
    
    %% 测试并保存结果
    RMSE_values = zeros(length(test_data), 1);
    for i = 1:length(test_data)
        test_sample = test_data{i};
        RMSE_values(i) = evaluate_lstm_lko(net, ...
                                          test_sample.control, ...
                                          test_sample.state, ...
                                          test_sample.label, ...
                                          params.delay_step);
    end
    
    mean_RMSE = mean(RMSE_values);
    fprintf('平均RMSE: %.4f\n', mean_RMSE);
    
    % 保存模型和结果
    save(fullfile(model_save_path, 'trained_network.mat'), 'net', 'A', 'B');
    save(fullfile(model_save_path, 'results.mat'), ...
         'params', 'RMSE_values', 'mean_RMSE');
    
    % 保存归一化参数（如果需要）
    % save(fullfile(model_save_path, 'norm_params.mat'), 'norm_params');
end

disp('所有参数组合训练完成！');