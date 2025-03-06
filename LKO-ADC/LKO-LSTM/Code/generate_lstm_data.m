% 参数设置
m = 3;        % 连续时刻数（示例值）  
data_folder = '../Data/BellowData/RawData/trainData'; 
save_path = '../Data/BellowData/lstmData/trainData/lstm_training_data.mat';
control_var_name = 'U_list'; 
state_var_name = 'X_list';     

% 获取所有.mat文件列表
file_list = dir(fullfile(data_folder, '*.mat'));
num_files = length(file_list);

% 初始化三维存储数组
control_sequences = [];  % c x m x N
state_sequences = [];    % d x m x N
label_sequences = [];   % d x m x N

for file_idx = 1:num_files
    % 加载数据
    file_path = fullfile(data_folder, file_list(file_idx).name);
    data = load(file_path);
    
    % 提取数据并验证维度
    control = data.(control_var_name);  % c x t
    states = data.(state_var_name);     % d x t
    [c, t] = size(control);
    [d, t_check] = size(states);
    
    % 数据一致性检查
    if t ~= t_check
        fprintf('跳过文件 %s（时间步不匹配）\n', file_list(file_idx).name);
        continue;
    end
    if t < m+1
        fprintf('跳过文件 %s（时间步不足）\n', file_list(file_idx).name);
        continue;
    end
    % 计算本文件样本数
    num_samples = t - m;  % 保证标签窗口有足够数据
    
    % 预分配本文件数据
    file_control = zeros(c,  num_samples);
    file_state = zeros(d, num_samples, m);
    file_labels = zeros(d, num_samples, m);
    
    % 构建时间窗口
    for sample_idx = 1:num_samples
        time_window = sample_idx : sample_idx + m - 1;
        
        % 控制输入序列 [p(t) ... p(t+m-1)]
        file_control(:, sample_idx) = control(:, sample_idx + m - 1);
        
        % 当前状态序列 [s(t) ... s(t+m-1)]
        file_state(:, sample_idx, :) = states(:, time_window);
        
        % 标签序列 [s(t+1) ... s(t+m)]
        file_labels(:,  sample_idx, :) = states(:, time_window + 1);
    end
    
    % 合并数据
    control_sequences = cat(2, control_sequences, file_control);
    state_sequences = cat(2, state_sequences, file_state);
    label_sequences = cat(2, label_sequences, file_labels);
end

% 保存结果
save(save_path, 'control_sequences', 'state_sequences', 'label_sequences');

% 显示信息
disp('最终数据维度:');
disp(['控制输入：', num2str(size(control_sequences))]);
disp(['状态输入：', num2str(size(state_sequences))]);
disp(['标签数据：', num2str(size(label_sequences))]);