% 参数设置
time_step = 3;
target_dimensions = 64;
lift_function = @embedding_polynomial;

% 设置路径
train_path = '../Data/BellowData/RawData/trainData';
test_path = '../Data/BellowData/RawData/testData';

% 获取所有.mat文件列表
file_list = dir(fullfile(train_path, '*.mat'));
num_files = length(file_list);

% 初始化三维存储数组
control_sequences = [];  % c x N
state_sequences = [];    % dm x N
label_sequences = [];   % dm x N

% 加载训练数据，处理为时间延迟数据
for file_idx = 1:num_files
    % 加载数据
    file_path = fullfile(train_path, file_list(file_idx).name);
    data = load(file_path);
    
    [file_control, file_state, file_labels] = generate_timeDelay_data(data, time_step);
    
    % 合并数据
    control_sequences = cat(2, control_sequences, file_control);
    state_sequences = cat(2, state_sequences, file_state);
    label_sequences = cat(2, label_sequences, file_labels);
end

% 归一化数据
[norm_control, params_control] = normalize_data(control_sequences);
[norm_state, params_state] = normalize_data(state_sequences);
[norm_label, params_label] = normalize_data(label_sequences);

% 显示信息
disp('最终数据维度:');
disp(['控制输入：', num2str(size(norm_control))]);
disp(['状态输入：', num2str(size(norm_state))]);
disp(['标签数据：', num2str(size(norm_label))]);

% 计算Koopman算子
[A, B] = koopman_operator(norm_control, norm_state, norm_label, lift_function, target_dimensions);

% 加载测试集数据，处理为时间延迟数据