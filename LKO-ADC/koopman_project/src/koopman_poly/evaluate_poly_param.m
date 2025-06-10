% 获取当前文件所在目录
currentDir = fileparts(mfilename('fullpath'));

% 获取上一级目录
parentDir = fileparts(currentDir);

% 只添加上一级目录本身（不包括其子目录）
addpath(parentDir);

%% 参数配置
% 搜索范围
delay_range = 1:10;          % delay_time搜索范围
dimension_range = 24;       % target_dimensions搜索范围
lift_func = @polynomial_expansion;

% 路径设置
train_path = fullfile('..\..\data', 'SorotokiData', 'MotionData5', 'FilteredDataPos', '80minTrain');
test_path = fullfile('..\..\data', 'SorotokiData', 'MotionData5', 'FilteredDataPos', '50secTest');
control_var_name = 'input'; 
state_var_name = 'state';    

% 预测参数
state_window = 1:6;
predict_window = 1:100;
lambda = 1e-4 ;  % 正则化系数

%% 初始化数据结构
% 生成参数组合
[delay_grid, dim_grid] = meshgrid(delay_range, dimension_range);
param_combos = [delay_grid(:), dim_grid(:)];
total_combos = size(param_combos,1);

% 预分配结果存储
results = struct(...
    'delay', num2cell(zeros(total_combos,1)),...
    'dimension', num2cell(zeros(total_combos,1)),...
    'test_rmse', cell(total_combos,1),...
    'mean_rmse', num2cell(zeros(total_combos,1)),...
    'std_rmse', num2cell(zeros(total_combos,1))...
);

% 加载测试文件列表
test_files = dir(fullfile(test_path, '*.mat'));
num_test_files = length(test_files);

%% 主参数循环
for combo_idx = 1:total_combos
    current_delay = param_combos(combo_idx,1);
    current_dim = param_combos(combo_idx,2);
    
    fprintf('\n=== 正在评估 [%d/%d] delay=%d, dim=%d ===\n',...
        combo_idx, total_combos, current_delay, current_dim);
    
    try
        %% 训练阶段
        % 加载训练数据
        train_files = dir(fullfile(train_path, '*.mat'));
        num_train_files = length(train_files);
        
        % 初始化训练数据存储
        train_control = [];
        train_state = [];
        
        % 合并所有训练轨迹
        for train_idx = 1:num_train_files
            file_data = load(fullfile(train_path, train_files(train_idx).name));
            train_control = cat(2, train_control, file_data.(control_var_name));
            train_state = cat(2, train_state, file_data.(state_var_name));
        end
        
        % 生成时间延迟数据
        [control_td, state_td, label_td] = generate_timeDelay_data(...
            train_control, train_state, current_delay);
        
        % 提升维度
        state_phi = lift_func(state_td, current_dim, current_delay);
        label_phi = lift_func(label_td, current_dim, current_delay);
        
        % 计算Koopman算子（带正则化）
        [A, B] = koopman_operator(control_td, state_phi, label_phi);
        
        %% 测试阶段
        test_rmse = zeros(num_test_files,1);
        detailed_log = cell(num_test_files,3);  % [文件名, RMSE, 数据长度]
        
        for test_idx = 1:num_test_files
            % 加载测试数据
            test_file = fullfile(test_path, test_files(test_idx).name);
            test_data = load(test_file);
            
            % 提取当前轨迹
            test_control = test_data.(control_var_name);
            test_state = test_data.(state_var_name);
            
            % 生成时间延迟
            [control_test, state_test, label_test] = generate_timeDelay_data(...
                test_control, test_state, current_delay);
            
            % 验证数据长度
            min_length = size(control_test,2);
            if min_length < current_delay + 1
                error('测试数据长度不足');
            end
            
            % 提升维度
            state_test_phi = lift_func(state_test, current_dim, current_delay);
            
            % 执行多步预测
            Y_true = label_test(state_window, predict_window + 30 - current_delay);
            Y_pred = predict_multistep(...
                A, B,...
                control_test(:, predict_window + 30 - current_delay),...
                state_test_phi(:, predict_window(1) + 30 - current_delay),...
                predict_window(end) - predict_window(1) + 1);
            Y_pred = Y_pred(state_window, :);
            
            % 计算RMSE
            current_rmse = calculateRMSE(Y_pred, Y_true);
            test_rmse(test_idx) = current_rmse;
            
            % 记录详细信息
            detailed_log(test_idx,:) = {...
                test_files(test_idx).name,...
                current_rmse,...
                size(Y_true,2)...
            };
            
            % 实时输出
            fprintf('  测试案例 %02d/%02d: %-20s RMSE=%.4f 数据点=%d\n',...
                test_idx, num_test_files,...
                test_files(test_idx).name,...
                current_rmse,...
                size(Y_true,2));
        end
        
        %% 存储结果
        results(combo_idx).delay = current_delay;
        results(combo_idx).dimension = current_dim;
        results(combo_idx).test_rmse = detailed_log;
        results(combo_idx).mean_rmse = mean(test_rmse);
        results(combo_idx).std_rmse = std(test_rmse);
        
        % 定期保存临时结果
        if mod(combo_idx,5) == 0
            save(fullfile('temp_results.mat'), 'results', '-v7.3');
        end
        
    catch ME
        fprintf('评估失败: %s\n', ME.message);
        results(combo_idx).delay = current_delay;
        results(combo_idx).dimension = current_dim;
        results(combo_idx).test_rmse = {'ERROR', ME.message, 0};
    end
end

%% 结果后处理
% 过滤有效结果

% 按平均RMSE降序排序
[~, sort_idx] = sort([results.mean_rmse], 'descend');
sorted_results = results(sort_idx);

% 生成表格输出
report_table = struct2table(sorted_results, 'AsArray',true);
writetable(report_table, fullfile('performance_report.csv'));

% 保存完整结果
save(fullfile('full_results.mat'), 'sorted_results', '-v7.3');

%% 最终输出
fprintf('\n======= i参数组合 =======\n');
disp(report_table(:, {'delay','dimension','mean_rmse','std_rmse'}));
diary off;

%% Koopman算子计算函数（带正则化）
function [A, B] = compute_koopman(U, X_phi, Y_phi)
    Psi = [X_phi; U];
    reg_matrix = 0*lambda * eye(size(Psi,1));
    AB = Y_phi * Psi' / (Psi * Psi' + reg_matrix);
    
    state_dim = size(X_phi,1);
    A = AB(:, 1:state_dim);
    B = AB(:, state_dim+1:end);
end