mainFolder = fileparts(mfilename('fullpath'));
% 添加主文件夹及其所有子文件夹到路径
addpath(genpath(mainFolder));
%% 参数设置
% 生成数据参数
time_step = 3;
loss_pred_step = 5;
control_var_name = 'inputdata'; 
state_var_name = 'x_meas';    
% 神经网络参数
params = struct();
params.feature_size = 3;                % 特征维度
params.node_size = 6;                   % 节点个数
params.adjMatrix = [0,1,0,1,0,0;1,0,1,0,1,0;0,1,0,0,0,1;
        1,0,0,0,1,0;0,1,0,1,0,1;0,0,1,0,1,0];  
params.adjMatrix = params.adjMatrix + eye(size(params.adjMatrix, 1));    % 添加自环
D = diag([sum(params.adjMatrix, 2)]);                                    % 度矩阵
params.adjMatrix = sqrt(inv(D))*params.adjMatrix*sqrt(inv(D));           % 对称归一化处理
params.control_size = 6;                % 控制输入维度
params.hidden_size = 32;               % 隐藏层维度
params.PhiDimensions = 68;              % 高维特征维度
params.output_size = params.PhiDimensions - params.feature_size*params.node_size;
params.initialLearnRate = 1e-2;         % 初始学习率
params.minLearnRate = 0;                % 最低学习率
params.num_epochs = 10000;                % 训练轮数
params.L1 = 100;                        % 损失权重1
params.L2 = 1;                        % 损失权重2
params.L3 = 0;                       % 损失权重3
params.batchSize = 8172*4;           % 批处理大小
params.patience = 50;            % 新增参数
params.lrReduceFactor = 0.2; % 新增参数

train_path = 'data\6Dof_FilteredDirection\trainData';
test_path = 'data\6Dof_FilteredDirection\testData';
model_save_path = ['models\LKO_GCN_directiondata_delay' num2str(time_step) 'pred' num2str(loss_pred_step) ...
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

% 初始化三维存储数组
control_sequences = [];  % c x N
state_sequences = [];    % dm x N

% 处理数据
for file_idx = 1:num_files
    % 加载数据
    file_path = fullfile(train_path, file_list(file_idx).name);
    data = load(file_path);
    % 合并数据
    control_sequences = cat(2, control_sequences, data.(control_var_name));
    state_sequences = cat(2, state_sequences, data.(state_var_name));
end

% 归一化数据
[norm_control, params_control] = normalize_data(control_sequences);
[norm_state, params_state] = normalize_data(state_sequences);
save([model_save_path, 'norm_params'], 'params_state', 'params_control')

% 生成时间延迟数据
[control_timedelay, state_timedelay, label_timedelay] = ...
    generate_gcn_data(norm_control, norm_state, time_step, loss_pred_step); 
%% 训练
train_data.control_sequences = control_timedelay;
train_data.state_sequences = state_timedelay;
train_data.label_sequences = label_timedelay;

[last_model, A, B] = train_gcn_lko(params, train_data, model_save_path);


