load('best_net.mat');
net = best_net;
test_path = '..\..\data\SorotokiData\MotionData7\FilteredData\50secTest';
train_path = '..\..\data\SorotokiData\MotionData7\FilteredData\80minTrain';
is_norm = true;
delay_step = 7;

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


%% 加载训练数据
% 获取所有.mat文件列表
file_list = dir(fullfile(train_path, '*.mat'));
num_files = length(file_list);

control_train = [];
label_train = [];
state_train = [];

state_for_norm = [];

for file_idx = 1:num_files
    % 加载数据
    file_path = fullfile(train_path, file_list(file_idx).name);
    data = load(file_path);
    % 合并数据
    state_for_norm = cat(2, state_for_norm, data.(state_var_name));
end
[~, params_state] = normalize_data(state_for_norm);



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

    if is_norm
        state_test = normalize_data(state_test, params_state);
    end

    % 生成时间延迟数据
    [control_timedelay_test, state_timedelay_test, label_timedelay_test] = ...
        generate_lstm_data(control_test, state_test, params.delay_step, loss_pred_step); 
    

    current_test_data = struct('control', control_timedelay_test, 'state', state_timedelay_test, ...
        'label', label_timedelay_test);
        

    test_data{file_idx} = current_test_data;

end


% 遍历每个测试文件
for i = 1:numel(test_data)
    % 加载单个测试文件
    
    % 提取当前轨迹数
    
    control_test = test_data{i}.control;
    state_test = test_data{i}.state;
    label_test = test_data{i}.label;
    [~, Y_true, Y_pred] = evaluate_lstm_lko(net, control_test, state_test, label_test, params.delay_step);
    
    % if is_norm
    %     Y_pred = denormalize_data(Y_pred, params_state);
    %     Y_true = denormalize_data(Y_true, params_state);
    % end

    RMSE(i) = calculateRMSE(Y_true, Y_pred);
    
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
fprintf('RMSE损失 %f \n', mean(RMSE));