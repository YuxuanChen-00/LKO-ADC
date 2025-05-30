mainFolder = fileparts(mfilename('fullpath'));
addpath(genpath(mainFolder));

%% 基础参数设置
control_var_name = 'input'; 
state_var_name = 'state';    
loss_pred_step = 1;
model_save_path = 'models\LKO_LSTM_SorotokiPositionData_network\';
train_path = 'data\SorotokiData\MotionData4\FilteredDataPos\40minTrain';
test_path = 'data\SorotokiData\MotionData4\FilteredDataPos\50secTest';

% 创建模型保存路径
if ~exist(model_save_path, 'dir')
    mkdir(model_save_path);
end

%% 参数迭代范围
delay_steps = 3:10;
phi_dimensions = 10:5:30;

% 初始化结果记录
all_results = cell(length(delay_steps)*length(phi_dimensions), 4);
result_index = 1;

%% 参数迭代主循环
for delay_step = delay_steps
    for phi_dim = phi_dimensions
        fprintf('\n========= 训练参数组合: delay_step=%d, PhiDimensions=%d =========\n',...
                delay_step, phi_dim);
            
        %% 动态参数配置
        params = struct();
        params.state_size = 6;                % 特征维度
        params.delay_step = delay_step;       % 当前delay step
        params.control_size = 6;              % 控制输入维度
        params.hidden_size = 16;              % 隐藏层维度
        params.PhiDimensions = phi_dim;        % 当前phi维度
        params.output_size = phi_dim - params.state_size;
        params.initialLearnRate = 4e-3;       % 初始学习率
        params.minLearnRate = 0;               % 最低学习率
        params.num_epochs = 1000;               % 训练轮数
        params.L1 = 1;                      % 损失权重1
        params.L2 = 1;                        % 损失权重2
        params.L3 = 0;                         % 损失权重3
        params.batchSize = 128;               % 批处理大小
        params.patience = 20;                  % early stopping耐心值
        params.lrReduceFactor = 0.2;           % 学习率衰减因子

        %% 数据预处理
        % 训练数据处理
        train_file_list = dir(fullfile(train_path, '*.mat'));
        control_train = [];
        state_train = [];
        label_train = [];
        
        for file_idx = 1:length(train_file_list)
            file_path = fullfile(train_path, train_file_list(file_idx).name);
            data = load(file_path);
            
            % 生成时间延迟数据（使用当前delay_step）
            [control_td, state_td, label_td] = generate_lstm_data(...
                data.(control_var_name),...
                data.(state_var_name),...
                delay_step,...
                loss_pred_step);
            
            control_train = cat(2, control_train, control_td);
            state_train = cat(2, state_train, state_td);
            label_train = cat(2, label_train, label_td);
        end
        
        train_data = struct(...
            'control_sequences', control_train,...
            'state_sequences', state_train,...
            'label_sequences', label_train);
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
        %% 模型训练
        [net, A, B] = train_lstm_lko(params, train_data, test_data);  
        
        %% 模型评估
        test_rmse = zeros(numel(test_data), 1);
        for i = 1:numel(test_data)
            control_test = test_data{i}.control;
            state_test = test_data{i}.state;
            label_test = test_data{i}.label;
            test_rmse(i) = evaluate_lstm_lko(net, control_test, state_test, label_test, params.delay_step);
        end    
        avg_rmse = mean(test_rmse);

        %% 保存结果
        % 保存模型参数
        model_name = sprintf('delay%d_phi%d_rmse%.4f.mat',...
                           delay_step, phi_dim, avg_rmse);
        save(fullfile(model_save_path, model_name),...
            'net', 'A', 'B', 'params', 'avg_rmse');
        
        % 记录结果
        all_results(result_index,:) = {delay_step, phi_dim, avg_rmse, model_name};
        result_index = result_index + 1;
        
        % 输出当前结果
        fprintf('当前参数组合 RMSE: %.4f\n', avg_rmse);
        fprintf('模型已保存为: %s\n\n', model_name);
    end
end

%% 保存完整结果记录
results_table = cell2table(all_results,...
    'VariableNames', {'DelayStep', 'PhiDim', 'RMSE', 'ModelName'});
writetable(results_table, fullfile(model_save_path, 'training_results.csv'));

fprintf('所有参数组合训练完成！结果已保存至: %s\n', model_save_path);