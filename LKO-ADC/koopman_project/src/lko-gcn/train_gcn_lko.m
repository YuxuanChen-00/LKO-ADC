function [net, A, B] = train_gcn_lko(params, train_data, model_savePath)
    %% 参数加载
    feature_size = params.feature_size;
    node_size = params.node_size;
    adjMatrix = params.adjMatrix;
    control_size = params.control_size;
    hidden_size = params.hidden_size;
    output_size = params.output_size;
    initialLearnRate = params.initialLearnRate;
    minLearnRate = params.minLearnRate;
    num_epochs = params.num_epochs;
    L1 = params.L1;
    L2 = params.L2;
    L3 = params.L3;
    batchSize = params.batchSize;
    restart_times = params.restart_times;
    num_stages = restart_times + 1;
    stage_epochs = split_epochs(num_epochs, num_stages); % 分配各阶段epoch数

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
    num_samples = size(control_sequences, 3);
    shuffled_idx = randperm(num_samples);
    % 计算分割点
    test_ratio = 0.2;    % 测试集比例
    split_point = floor(num_samples * (1 - test_ratio));
    % 训练集和测试集索引
    train_idx = shuffled_idx(1:split_point);
    test_idx = shuffled_idx(split_point+1:end);
    % 提取数据
    control_train = control_sequences(:, :, train_idx);
    state_train = state_sequences(:, :, :, train_idx);
    label_train = label_sequences(:, :, :, train_idx);
    
    control_test = control_sequences(:, :, test_idx);
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
    net = lko_gcn_network(feature_size, node_size, adjMatrix, hidden_size,output_size, control_size);
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
    num_iterations_per_epoch = floor(numTrainingInstances / batchSize); % 每个epoch迭代次数
    
   %% 分阶段训练
    current_epoch = 1;
    for stage = 1:num_stages
        stage_num_epochs = stage_epochs(stage);
        stage_T_max = stage_num_epochs * num_iterations_per_epoch; % 当前阶段总迭代次数
        
        %% 初始化优化器状态（热重启关键步骤）
        averageGrad = [];
        averageSqGrad = [];
        iteration = 0; % 阶段内迭代计数器

        %% 阶段内训练循环
        for epoch_in_stage = 1:stage_num_epochs
            epoch = current_epoch + epoch_in_stage - 1;
            
            %% 重置数据存储（原epoch循环内容）
            mbq_train = minibatchqueue(ds_train, ...
                'MiniBatchSize', batchSize, ...
                'MiniBatchFcn', @preprocessMiniBatch, ...
                'OutputEnvironment', device, ...
                'PartialMiniBatch', 'discard');
            mbq_test = minibatchqueue(ds_test, ...
                'MiniBatchSize', batchSize, ...
                'MiniBatchFcn', @preprocessMiniBatch, ...
                'OutputEnvironment', device, ...
                'PartialMiniBatch', 'discard');

            %% 训练步骤
            while hasdata(mbq_train)
                [control, state, label] = next(mbq_train);
                
                % 计算带热重启的余弦退火学习率
                cos_lr = minLearnRate + 0.5*(initialLearnRate - minLearnRate)*...
                        (1 + cos(pi * iteration / stage_T_max));
                iteration = iteration + 1;
                % 梯度计算与参数更新
                [total_loss, gradients] = dlfeval(@modelGradients, net, state, control, label, L1, L2, L3, feature_size, node_size);
                [net, averageGrad, averageSqGrad] = adamupdate(net, gradients, averageGrad, averageSqGrad, iteration, cos_lr);
                
            end

            %% 测试步骤
            test_epoch_iteration = 0;
            test_loss = 0;
            while hasdata(mbq_test)
                [control, state, label] = next(mbq_test);
                current_test_loss = gcn_loss_function2(net, state, control, label, L1, L2, L3, feature_size, node_size);
                test_loss = test_loss + current_test_loss;
                test_epoch_iteration = test_epoch_iteration + 1;
            end
            test_loss = test_loss / test_epoch_iteration;
            
            %% 日志输出
            fprintf('Stage %d, Epoch %d, 训练损失: %.4f, 测试损失: %.4f, 学习率: %.4f\n',...
                    stage, epoch, total_loss, test_loss, cos_lr);
            
            %% 模型保存（保持原有逻辑）
            if mod(epoch, 10) == 0
                save([model_savePath, 'gcn_network_epoch',num2str(epoch),'.mat'], 'net');
                A = net.Layers(8).Weights;
                B = net.Layers(9).Weights;
                save([model_savePath, 'gcn_KoopmanMatrix_epoch',num2str(epoch),'.mat'], 'A', 'B');
            end
        end
        current_epoch = current_epoch + stage_num_epochs;
    end
    
    disp('训练完成，网络和矩阵已保存！');


    function [controls, states, labels] = preprocessMiniBatch(controlCell, stateCell, labelCell)
        % 处理control数据（格式转换：CB）
        controls = cat(3, controlCell{:});  % 合并为 [特征数 x batchSize]
        controls = dlarray(controls, 'SCB'); % 转换为dlarray并指定格式
        
        % 处理state和label数据（格式转换：CBT）
        % 获取维度信息
        numFeatures = size(stateCell{1}, 1);
        numNodes = size(stateCell{1}, 2);
    
        % 合并并重塑state数据
        states = cat(4, stateCell{:});  % 合并为 [特征数 x (batchSize*numTimeSteps)]
        states = reshape(states, numFeatures, numNodes, 1, []); % [特征数 x batchSize x 时间步]
        states = dlarray(states, 'SSCB');
        
        % 对label执行相同操作
        labels = cat(4, labelCell{:});
        labels = reshape(labels, numFeatures, numNodes, [], size(labels, 4));
        labels = dlarray(labels, 'SSCB');

    end
    function [total_loss, gradients] = modelGradients(net, state, control, label, L1, L2,L3, feature_size, node_size)
        % 前向传播获取预测值
        total_loss = gcn_loss_function2(net, state, control, label, L1, L2, L3, feature_size, node_size);
        % 计算梯度并梯度裁剪
        gradients = dlgradient(total_loss, net.Learnables);
    end

    % 自定义epoch分配函数
    function stage_epochs = split_epochs(total_epochs, num_stages)
        base_epochs = floor(total_epochs / num_stages);
        remainder = mod(total_epochs, num_stages);
        stage_epochs = ones(1, num_stages) * base_epochs;
        stage_epochs(1:remainder) = stage_epochs(1:remainder) + 1;
    end
end