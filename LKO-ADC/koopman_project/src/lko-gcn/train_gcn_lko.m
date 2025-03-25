function [net, A, B] = train_gcn_lko(train_data, model_savePath)
    %% 参数设置
    feature_size = 6;         % 特征维度
    node_size = 6;            % 节点个数
    adjMatrix = [0,1,0,1,0,0;1,0,1,0,1,0;0,1,0,0,0,1;
        1,0,0,0,1,0;0,1,0,1,0,1;0,0,1,0,1,0];           % 邻接矩阵
    adjMatrix = adjMatrix + eye(size(adjMatrix, 1));    % 添加自环
    D = diag([sum(adjMatrix, 2)]);                      % 度矩阵
    adjMatrix = sqrt(inv(D))*adjMatrix*sqrt(inv(D));    % 对称归一化处理

    time_step = 3;
    control_size = 6;        % 控制输入维度（根据您的数据调整）
    hidden_size = 64;        % 隐藏层维度
    PhiDimensions = 68;      % 拼接后的高维特征维度
    output_size = PhiDimensions-feature_size*node_size;        % phi的维度
    initialLearnRate = 1e-2;% 初始学习率
    minLearnRate = 0;        % 最低学习率
    num_epochs = 100;        % 训练轮数
    L1 = 100;                % 状态预测损失权重
    L2 = 100;                % 高维状态预测损失权重
    L3 = 0.01;
    batchSize = 16344;
    
    
    %% 检查GPU可用性并初始化
    useGPU = canUseGPU();  % 自定义函数检查GPU可用性
    if useGPU
        disp('检测到可用GPU，启用加速');
        device = 'gpu';
    else
        disp('未检测到GPU，使用CPU');
        device = 'cpu';
    end

    %% 训练数据加载
    fields = fieldnames(train_data);
    control_sequences = train_data.(fields{1});
    state_sequences = train_data.(fields{2});
    label_sequences = train_data.(fields{3});
    
    % 随机打乱索引
    num_samples = size(control_sequences, 2);
    shuffled_idx = randperm(num_samples);
    % 计算分割点
    test_ratio = 0.2;    % 测试集比例
    split_point = floor(num_samples * (1 - test_ratio));
    % 训练集和测试集索引
    train_idx = shuffled_idx(1:split_point);
    test_idx = shuffled_idx(split_point+1:end);
    % 提取数据
    control_train = control_sequences(:, train_idx);
    state_train = state_sequences(:, :, :, train_idx);
    label_train = label_sequences(:, :, :, train_idx);
    
    control_test = control_sequences(:, test_idx);
    state_test = state_sequences(:, :, :, test_idx);
    label_test = label_sequences(:, :, :, test_idx);
    
    % 训练集数据存储
    trainControlDatastore = arrayDatastore(control_train, 'IterationDimension', 3);
    trainStateDatastore = arrayDatastore(state_train, 'IterationDimension', 4);
    trainLabelDatastore = arrayDatastore(label_train, 'IterationDimension', 4);
    ds_train = combine(trainControlDatastore, trainStateDatastore, trainLabelDatastore);
    ds_train = shuffle(ds_train); % 训练集打乱
    
    % 测试集数据存储
    testControlDatastore = arrayDatastore(control_test, 'IterationDimension', 3);
    testStateDatastore = arrayDatastore(state_test, 'IterationDimension', 4);
    testLabelDatastore = arrayDatastore(label_test, 'IterationDimension', 4);
    ds_test = combine(testControlDatastore, testStateDatastore, testLabelDatastore);

    
    %% 网络初始化
    net = lko_gcn_network(feature_size, node_size, adjMatrix, hidden_size,output_size, control_size, time_step);
    net = net.Net;
    
    fprintf('\n详细层索引列表:\n');
    for i = 1:numel(net.Layers)
        % 显示层索引、层名称和层类型
        fprintf('Layer %2d: %-20s (%s)\n',...
            i,...
            net.Layers(i).Name,...
            class(net.Layers(i)));
    end

    %% 训练设置
    % 计算总迭代次数（T_max）
    numTrainingInstances = num_samples; % 训练样本总数
    numIterationsPerEpoch = floor(numTrainingInstances / batchSize);
    T_max = num_epochs * numIterationsPerEpoch; % 总迭代次数
    % 初始化优化器状态和迭代计数器
    averageGrad = [];
    averageSqGrad = [];
    iteration = 0;
    
    %% 自定义训练循环
    for epoch = 1:num_epochs
        % 重置数据存储并重新生成minibatchqueue
        mbq_train = minibatchqueue(ds_train, ...
            'MiniBatchSize', batchSize, ...
            'MiniBatchFcn', @preprocessMiniBatch, ...
            'OutputEnvironment', device, ...  % 自动选择运行环境（CPU/GPU）
            'PartialMiniBatch', 'discard');   % 不足一个batch时丢弃
        mbq_test = minibatchqueue(ds_test, ...
        'MiniBatchSize', batchSize, ...
        'MiniBatchFcn', @preprocessMiniBatch, ...
        'OutputEnvironment', device, ...  % 自动选择运行环境（CPU/GPU）
        'PartialMiniBatch', 'discard');   % 不足一个batch时丢弃
    
        % 训练
        while hasdata(mbq_train)
            % 获取当前批次数据
            [control, state, label] = next(mbq_train);
            

             % 使用dlfeval封装梯度计算
            [total_loss, gradients] = dlfeval(@modelGradients, net, state, control, label, L1, L2,L3, feature_size, node_size);
            
            % 计算余弦退火学习率
            cos_lr = minLearnRate + 0.5*(initialLearnRate - minLearnRate)*(1 + cos(pi * iteration / T_max));
            iteration = iteration + 1;
    
            % 更新参数（Adam优化器）
            [net, averageGrad, averageSqGrad] = adamupdate(net, gradients, averageGrad, averageSqGrad, iteration, cos_lr);
        end
    
        % 测试
        test_epoch_iteration = 0;
        test_loss = 0;
        while hasdata(mbq_test)
            [control, state, label] = next(mbq_test);
            % 前向传播获取预测值
            current_test_loss = gcn_loss_function(net, state, control, label, L1, L2, L3, feature_size, node_size);
            test_loss = test_loss + current_test_loss;
            test_epoch_iteration = test_epoch_iteration + 1;
        end
        test_loss = test_loss/test_epoch_iteration;
    
    
        fprintf('Epoch %d, 训练集当前损失: %.4f, 测试集当前损失: %.4f\n', epoch, total_loss, test_loss);
    
        % 保存网络和Koopman算子
        if mod(epoch, 10) == 0
            save([model_savePath, 'gcn_network_epoch',num2str(epoch),'.mat'], 'net');  % 保存整个网络
            A = net.Layers(7).Weights;  % 提取矩阵A
            B = net.Layers(8).Weights;  % 提取矩阵B
            save([model_savePath, 'gcn_KoopmanMatrix_epoch',num2str(epoch),'.mat'], 'A', 'B');  % 保存A和B矩阵
        end
    end
    
    disp('训练完成，网络和矩阵已保存！');



    function [controls, states, labels] = preprocessMiniBatch(controlCell, stateCell, labelCell)
        % 处理control数据（格式转换：CB）
        controls = cat(2, controlCell{:});  % 合并为 [特征数 x batchSize]
        controls = dlarray(controls, 'SCB'); % 转换为dlarray并指定格式
        
        % 处理state和label数据（格式转换：CBT）
        % 获取维度信息
        numFeatures = size(stateCell{1}, 1);
        numNodes = size(stateCell{1}, 2);
    
        % 合并并重塑state数据
        states = cat(2, stateCell{:});  % 合并为 [特征数 x (batchSize*numTimeSteps)]
        states = reshape(states, numFeatures, numNodes, 1, []); % [特征数 x batchSize x 时间步]
        states = dlarray(states, 'SSCB');
        
        % 对label执行相同操作
        labels = cat(2, labelCell{:});
        labels = reshape(labels, numFeatures, numNodes, [], size(labels, 4));
        labels = dlarray(labels, 'SSCB');
    end
    function [total_loss, gradients] = modelGradients(net, state, control, label, L1, L2,L3, feature_size, node_size)
        % 前向传播获取预测值
        total_loss = gcn_loss_function(net, state, control, label, L1, L2, L3, feature_size, node_size);
        % 计算梯度并梯度裁剪
        gradients = dlgradient(total_loss, net.Learnables);
    end
end