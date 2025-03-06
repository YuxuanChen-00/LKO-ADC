%% 参数设置
state_size = 18;         % 状态维度
control_size = 7;        % 控制输入维度（根据您的数据调整）
hidden_size = 64;        % LSTM隐藏层维度
output_size = 64;        % phi的维度
initialLearnRate = 1e-3;% 初始学习率
minLearnRate = 0;        % 最低学习率
num_epochs = 100;        % 训练轮数
L1 = 1000;                % 状态预测损失权重
L2 = 1.0;                % 高维状态预测损失权重
batchSize = 16;


data_folder = '../Data/TrainData/training_data.mat';  % 替换为您的文件夹路径
control_var_name = 'control_sequences';    % 控制输入变量名（根据.mat文件修改）
state_var_name = 'state_sequences';        % 状态变量名（根据.mat文件修改）
label_var_name = 'label_sequences';        % 标签变量名（根据.mat文件修改）


%% 检查GPU可用性并初始化
useGPU = canUseGPU();  % 自定义函数检查GPU可用性
if useGPU
    disp('检测到可用GPU，启用加速');
    device = 'gpu';
else
    disp('未检测到GPU，使用CPU');
    device = 'cpu';
end

%% 网络定义
layers = [
    sequenceInputLayer(state_size, 'Name', 'state_input')  % 输入状态
    fullyConnectedLayer(hidden_size, 'OutputMode', 'last', 'Name', 'lstm1')  % 第一层LSTM
    tanhLayer('Name','tanh1')
    fullyConnectedLayer(output_size, 'Name', 'fc_phi')      % 全连接层生成phi
    % batchNormalizationLayer('Name', 'batchnorm2')
    tanhLayer('Name','tanh')
];


% 创建网络
net = dlnetwork(layers);

% 添加连接9
net = addLayers(net, featureInputLayer(control_size, 'Name', 'control_input'));  % 控制输入
net = addLayers(net, fullyConnectedLayer(time_step*state_size + output_size, 'Name', 'A', 'BiasLearnRateFactor', 0)); % 无偏置线性层A
net = addLayers(net, fullyConnectedLayer(time_step*state_size + output_size, 'Name', 'B', 'BiasLearnRateFactor', 0)); % 无偏置线性层B
net = addLayers(net, concatenationLayer(1, 2, 'Name', 'concat')); % 拼接state和phi
net = addLayers(net, additionLayer(2, 'Name','add')); % 相加A*Phi+B*control_input

% net = addLayers(net, functionLayer(@(X) squeeze(reshape(X, [], size(X, 1))), 'Name', 'reshape')); % 三维转二维数据
net = addLayers(net, functionLayer(@(X) dlarray(reshape(permute(stripdims(X), [3, 1, 2]), [], size(X, 2)),'CB'), 'Name', 'reshape', 'Formattable', true)); % 三维转二维数据

net = connectLayers(net, 'state_input', 'reshape'); 
net = connectLayers(net, 'reshape', 'concat/in1'); % 连接state到拼接层
net = connectLayers(net, 'tanh', 'concat/in2'); % 连接phi到拼接层
net = connectLayers(net, 'concat', 'A');   % 连接控制输入到线性层A
net = connectLayers(net, 'control_input', 'B');   % 连接控制输入到线性层B
net = connectLayers(net, 'A', 'add/in1');
net = connectLayers(net, 'B', 'add/in2');

% 显示网络结构
% analyzeNetwork(net);

% 或者显式初始化（推荐）
inputStateExample = dlarray(rand(state_size,1,time_step),'CBT'); % 示例输入
inputControlExample = dlarray(rand(control_size,1),'CB');
net = initialize(net, inputStateExample, inputControlExample);

%% 训练数据加载
train_data = load('../Data/TrainData/training_data.mat');  % 加载之前生成的数据

control_sequences = train_data.(control_var_name);
state_sequences = train_data.(state_var_name);
label_sequences = train_data.(label_var_name);

% 创建数据存储
stateDatastore= arrayDatastore(state_sequences, 'IterationDimension', 2); % 按样本迭代
controlDatastore = arrayDatastore(control_sequences, 'IterationDimension', 2); % 按样本迭代
labelDatastore = arrayDatastore(label_sequences, 'IterationDimension', 2);


% 合并数据存储
ds_combined = combine(controlDatastore, stateDatastore, labelDatastore);
ds_combined = shuffle(ds_combined); % 打乱数据顺序，增强泛化性

function [controls, states, labels] = preprocessMiniBatch(controlCell, stateCell, labelCell)
    % 处理control数据（格式转换：CB）
    controls = cat(2, controlCell{:});  % 合并为 [特征数 x batchSize]
    controls = dlarray(controls, 'CB'); % 转换为dlarray并指定格式
    
    % 处理state和label数据（格式转换：CBT）
    % 获取维度信息
    numFeatures = size(stateCell{1}, 1);
    numTimeSteps = size(stateCell{1}, 3);

    % 合并并重塑state数据
    states = cat(2, stateCell{:});  % 合并为 [特征数 x (batchSize*numTimeSteps)]
    states = reshape(states, numFeatures, [], numTimeSteps); % [特征数 x batchSize x 时间步]
    states = dlarray(states, 'CBT');
    
    % 对label执行相同操作
    labels = cat(2, labelCell{:});
    labels = reshape(labels, numFeatures, [], numTimeSteps);
    labels = dlarray(labels, 'CBT');
end
function [total_loss, gradients] = modelGradients(net, state, control, label, L1, L2, state_size, time_step)
    % 前向传播获取预测值
    Phi_pred = forward(net, state, control);  % 获取网络输出
    state_pred = Phi_pred(1:state_size*time_step, :);  % 提取预测状态
    Phi = forward(net, label, control, 'Outputs', 'concat');
    
    % L2正则化
    weights = net.Learnables.Value;
    l2Reg = sum(cellfun(@(w) sum(w.^2, 'all'), weights)); % 计算L2正则项
    
    % 计算损失
    loss_state = L1 * mse(state_pred, dlarray(reshape(permute(stripdims(label), [3, 1, 2]), [], size(label, 2)),'CB'));
    % loss_phi = L2 * mse(Phi_pred, Phi);
    % 
    % disp(loss_state)
    % 
    % total_loss = loss_state + loss_phi;



    total_loss =loss_state + l2Reg;
    % 计算梯度并梯度裁剪
    gradients = dlgradient(total_loss, net.Learnables);
    gradients = dlupdate(@(g) clipGradient(g, 2), gradients);
    function clipped_g = clipGradient(g, threshold)
    normVal = sqrt(sum(g.^2, 'all'));
    if normVal > threshold
        clipped_g = g * (threshold / normVal);
    else
        clipped_g = g;
    end
end
end
% 创建 minibatchqueue
mbq = minibatchqueue(ds_combined, ...
    'MiniBatchSize', batchSize, ...
    'MiniBatchFcn', @preprocessMiniBatch, ...
    'OutputEnvironment', 'auto', ...  % 自动选择运行环境（CPU/GPU）
    'PartialMiniBatch', 'discard');   % 不足一个batch时丢弃


%% 训练设置
% 计算总迭代次数（T_max）
numTrainingInstances = size(label_sequences, 2); % 训练样本总数
numIterationsPerEpoch = floor(numTrainingInstances / batchSize);
T_max = num_epochs * numIterationsPerEpoch; % 总迭代次数
% 初始化优化器状态和迭代计数器
averageGrad = [];
averageSqGrad = [];
iteration = 0;

break_flag = false;
%% 自定义训练循环
for epoch = 1:num_epochs
    while hasdata(mbq)
        if iteration == 158 || iteration == 159
            next(mbq);
            iteration = iteration+1;
            continue
        end
        % 获取当前批次数据
        [control, state, label] = next(mbq);
        
         % 使用dlfeval封装梯度计算
        [total_loss, gradients] = dlfeval(@modelGradients, net, state, control, label, L1, L2, state_size, time_step);
        

        % 计算余弦退火学习率
        cos_lr = minLearnRate + 0.5*(initialLearnRate - minLearnRate)*(1 + cos(pi * iteration / T_max));
        iteration = iteration + 1;

        % 更新参数（Adam优化器）
        [net, averageGrad, averageSqGrad] = adamupdate(net, gradients, averageGrad, averageSqGrad, iteration, cos_lr);
    
        disp ([total_loss, iteration]);
        if  isnan(total_loss)
            break_flag = true;
            break
        end
    end
    
    if break_flag
        fprintf('检测到跳出条件，终止所有循环\n');
        break
    end

    fprintf('Epoch %d, Loss: %.4f\n', epoch, total_loss);
end

% %% 保存网络和矩阵
% save('trained_network.mat', 'net');  % 保存整个网络
% A = net.Layers(6).Weights;  % 提取矩阵A
% B = net.Layers(7).Weights;  % 提取矩阵B
% save('linear_matrices.mat', 'A', 'B');  % 保存A和B矩阵
% 
% disp('训练完成，网络和矩阵已保存！');