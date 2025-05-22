%% 初始化设置
mainFolder = fileparts(mfilename('fullpath')); % 获取当前文件所在文件夹
% 添加主文件夹及其所有子文件夹到路径
addpath(genpath(mainFolder));

%% 参数设置 (迭代相关的除外)
lift_function = @polynomial_expansion_td; % 定义提升函数
% train_path = 'data\SorotokiData\Filtered_PositionData\trainData'; % 训练数据路径 (备选)
% test_path = 'data\SorotokiData\Filtered_PositionData\testData';   % 测试数据路径 (备选)
train_path = 'data\SorotokiData\MotionData3_without_Direction\trainData'; % 训练数据路径
test_path = 'data\SorotokiData\MotionData3_without_Direction\testData';   % 测试数据路径
control_var_name = 'input'; % .mat 文件中控制变量的名称
state_var_name = 'state';    % .mat 文件中状态变量的名称
state_window = 1:6;          % 用于评估RMSE的状态维度窗口 (例如，原始状态的前6个维度)
predict_window_base = 1:100; % 基础预测窗口的长度 (预测500个时间步)

%% 加载原始训练数据 (一次性加载，避免在循环中重复加载)
fprintf('正在加载训练数据...\n');
train_file_list = dir(fullfile(train_path, '*.mat')); % 获取训练文件夹下所有 .mat 文件
num_train_files = length(train_file_list);
raw_control_sequences_train = [];  % 初始化原始控制序列 (训练)
raw_state_sequences_train = [];    % 初始化原始状态序列 (训练)

for file_idx = 1:num_train_files
    file_path = fullfile(train_path, train_file_list(file_idx).name);
    data = load(file_path);
    raw_control_sequences_train = cat(2, raw_control_sequences_train, data.(control_var_name));
    raw_state_sequences_train = cat(2, raw_state_sequences_train, data.(state_var_name));
end
fprintf('训练数据加载完毕。\n');

%% 加载原始测试数据 (一次性加载)
fprintf('正在加载测试数据...\n');
test_file_list = dir(fullfile(test_path, '*.mat')); % 获取测试文件夹下所有 .mat 文件
num_test_files = length(test_file_list);
raw_control_sequences_test = [];  % 初始化原始控制序列 (测试)
raw_state_sequences_test = [];    % 初始化原始状态序列 (测试)

for file_idx = 1:num_test_files
    file_path = fullfile(test_path, test_file_list(file_idx).name);
    data = load(file_path);
    raw_control_sequences_test = cat(2, raw_control_sequences_test, data.(control_var_name));
    raw_state_sequences_test = cat(2, raw_state_sequences_test, data.(state_var_name));
end
fprintf('测试数据加载完毕。\n');

%% 参数迭代循环
delay_time_range = 1:10;         % delay_time 的迭代范围
target_dimensions_range = 6:20; % target_dimensions 的迭代范围

results = {}; % 初始化元胞数组，用于存储有效的RMSE结果 {delay_time, target_dimensions, RMSE}

fprintf('\n开始参数迭代...\n');
fprintf('--------------------------------------------------------\n');
fprintf('| delay_time | target_dimensions |        RMSE        |\n');
fprintf('--------------------------------------------------------\n');

for delay_time = delay_time_range
    % 检查 delay_time 对于数据长度是否有效
    min_seq_len_required = delay_time; 
    if size(raw_control_sequences_train, 2) < min_seq_len_required || size(raw_state_sequences_train, 2) < min_seq_len_required
        fprintf('delay_time = %d 太大，训练数据长度不足，跳过。\n', delay_time);
        continue;
    end
    if size(raw_control_sequences_test, 2) < min_seq_len_required || size(raw_state_sequences_test, 2) < min_seq_len_required
        fprintf('delay_time = %d 太大，测试数据长度不足，跳过。\n', delay_time);
        continue;
    end

    % 1. 为当前 delay_time 生成训练集的时间延迟数据
    [control_timedelay_train, state_timedelay_train, label_timedelay_train] = ...
        generate_timeDelay_data(raw_control_sequences_train, raw_state_sequences_train, delay_time);
    
    if isempty(control_timedelay_train) || isempty(state_timedelay_train) || isempty(label_timedelay_train)
        fprintf('delay_time = %d 时，generate_timeDelay_data (训练集) 返回空数组，跳过此 delay_time。\n', delay_time);
        continue;
    end

    % 2. 为当前 delay_time 生成测试集的时间延迟数据
    [control_timedelay_test, state_timedelay_test, label_timedelay_test] = ...
        generate_timeDelay_data(raw_control_sequences_test, raw_state_sequences_test, delay_time);

    if isempty(control_timedelay_test) || isempty(state_timedelay_test) || isempty(label_timedelay_test)
        fprintf('delay_time = %d 时，generate_timeDelay_data (测试集) 返回空数组，跳过此 delay_time。\n', delay_time);
        continue;
    end
    
    prediction_col_offset = 99; 
    indices_for_prediction_data = predict_window_base + prediction_col_offset - delay_time;
    initial_phi_col_index_in_td_data = predict_window_base(1) + prediction_col_offset - delay_time;
    num_prediction_steps = length(predict_window_base);

    if initial_phi_col_index_in_td_data < 1 || ...
       indices_for_prediction_data(1) < 1 || ...
       indices_for_prediction_data(end) > size(label_timedelay_test, 2) || ...
       indices_for_prediction_data(end) > size(control_timedelay_test, 2) || ...
       initial_phi_col_index_in_td_data > size(state_timedelay_test, 2)
        fprintf('delay_time = %d 导致预测索引无效，跳过此 delay_time。\n', delay_time);
        continue;
    end

    for target_dimensions = target_dimensions_range
        try
            state_timedelay_phi_train = lift_function(state_timedelay_train, target_dimensions, delay_time);
            label_timedelay_phi_train = lift_function(label_timedelay_train, target_dimensions, delay_time);
            
            if isempty(state_timedelay_phi_train) || isempty(label_timedelay_phi_train)
                fprintf('| %10d | %17d | %-18s |\n', delay_time, target_dimensions, 'LiftTrainDataEmpty');
                continue;
            end
            if size(state_timedelay_phi_train,2) ~= size(control_timedelay_train,2) || ...
               size(label_timedelay_phi_train,2) ~= size(control_timedelay_train,2)
                fprintf('| %10d | %17d | %-18s |\n', delay_time, target_dimensions, 'TrainColMismatch');
                continue;
            end

            [A, B] = koopman_operator(control_timedelay_train, state_timedelay_phi_train, label_timedelay_phi_train);

            state_timedelay_phi_test = lift_function(state_timedelay_test, target_dimensions, delay_time);

            if isempty(state_timedelay_phi_test)
                fprintf('| %10d | %17d | %-18s |\n', delay_time, target_dimensions, 'LiftTestDataEmpty');
                continue;
            end
            
            if initial_phi_col_index_in_td_data > size(state_timedelay_phi_test, 2)
                fprintf('| %10d | %17d | %-18s |\n', delay_time, target_dimensions, 'InitPhiIdxOOB');
                continue;
            end

            Y_true = label_timedelay_test(state_window, indices_for_prediction_data);
            control_input_for_prediction = control_timedelay_test(:, indices_for_prediction_data);
            initial_state_phi_for_prediction = state_timedelay_phi_test(:, initial_phi_col_index_in_td_data);
            
            Y_pred_lifted = predict_multistep(A, B, control_input_for_prediction, ...
                                 initial_state_phi_for_prediction, ...
                                 num_prediction_steps);
            
            if size(Y_pred_lifted, 1) < max(state_window)
                fprintf('| %10d | %17d | %-18s |\n', delay_time, target_dimensions, 'PredDimTooSmall');
                continue;
            end
            
            Y_pred = Y_pred_lifted(state_window, :);
            current_rmse = calculateRMSE(Y_pred, Y_true);
            
            if isnumeric(current_rmse) && isfinite(current_rmse)
                fprintf('| %10d | %17d | %18.6f |\n', delay_time, target_dimensions, current_rmse);
                results = [results; {delay_time, target_dimensions, current_rmse}]; % 记录有效结果
            else
                fprintf('| %10d | %17d | %18s |\n', delay_time, target_dimensions, 'Invalid RMSE');
            end

        catch ME
            fprintf('| %10d | %17d | ERROR: %-12.12s |\n', delay_time, target_dimensions, ME.identifier);
        end 
    end 
    if ~isempty(target_dimensions_range) % 只在内循环执行过时打印分隔线
        fprintf('--------------------------------------------------------\n');
    end
end 

fprintf('参数迭代完成。\n');

%% 排序并输出结果
if ~isempty(results)
    fprintf('\n\n按RMSE从大到小排序的结果:\n');
    fprintf('--------------------------------------------------------\n');
    fprintf('| delay_time | target_dimensions |        RMSE        |\n');
    fprintf('--------------------------------------------------------\n');
    
    % 提取RMSE值用于排序
    rmse_values_for_sorting = cell2mat(results(:,3));
    % 获取降序排序的索引
    [~, sorted_indices] = sort(rmse_values_for_sorting, 'descend');
    % 根据索引重排结果
    sorted_results = results(sorted_indices, :);
    
    for i = 1:size(sorted_results, 1)
        fprintf('| %10d | %17d | %18.6f |\n', sorted_results{i,1}, sorted_results{i,2}, sorted_results{i,3});
    end
    fprintf('--------------------------------------------------------\n');
    
    % (可选) 将排序后的结果保存到 .mat 文件或 .csv 文件
    % save('sorted_iteration_results.mat', 'sorted_results');
    % results_table = cell2table(sorted_results, 'VariableNames', {'DelayTime', 'TargetDimensions', 'RMSE_Value'});
    % writetable(results_table, 'sorted_iteration_results.csv');
    % disp(results_table); % 如果想在命令行显示表格形式的结果

else
    fprintf('\n没有有效的RMSE结果可以排序和显示。\n');
end