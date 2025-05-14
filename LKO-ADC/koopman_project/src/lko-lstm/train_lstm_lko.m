function [net, A, B] = train_lstm_lko(params, train_data, model_savePath)
    %% 参数设置
    state_size = params.state_size;
    time_step = params.time_step;
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
    patience = params.patience;            % 新增参数
    lrReduceFactor = params.lrReduceFactor; % 新增参数
    
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
    state_train = state_sequences(:, train_idx, :);
    label_train = label_sequences(:, :, train_idx, :);
    

    control_test = control_sequences(:, :,test_idx);
    state_test = state_sequences(:, test_idx, :);
    label_test = label_sequences(:, :, test_idx,:);
    
    % 训练集数据存储
    trainControlDatastore = arrayDatastore(control_train, 'IterationDimension', 3);
    trainStateDatastore = arrayDatastore(state_train, 'IterationDimension', 2);
    trainLabelDatastore = arrayDatastore(label_train, 'IterationDimension', 3);
    ds_train = combine(trainControlDatastore, trainStateDatastore, trainLabelDatastore);
    ds_train = shuffle(ds_train); % 训练集打乱
    
    % 测试集数据存储
    testControlDatastore = arrayDatastore(control_test, 'IterationDimension', 3);
    testStateDatastore = arrayDatastore(state_test, 'IterationDimension', 2);
    testLabelDatastore = arrayDatastore(label_test, 'IterationDimension', 3);
    ds_test = combine(testControlDatastore, testStateDatastore, testLabelDatastore);


    
    %% 网络初始化
    net = lko_lstm_network(state_size, hidden_size, output_size, control_size, time_step);
    net = net.Net;
    analyzeNetwork(net)
    fprintf('\n详细层索引列表:\n');

    for i = 1:numel(net.Layers)
        % 显示层索引、层名称和层类型
        fprintf('Layer %2d: %-20s (%s)\n',...
            i,...
            net.Layers(i).Name,...
            class(net.Layers(i)));
    end
    %% 训练设置
    averageGrad = [];
    averageSqGrad = [];
    iteration = 0;

    % 初始化学习率调度状态
    best_test_loss = Inf;    % 最佳验证损失
    wait_counter = 0;        % 无改善计数器
    current_lr = initialLearnRate; % 当前学习率
    
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
            [total_loss, gradients] = dlfeval(@modelGradients, net, state, control, label, L1, L2, state_size, time_step);
            iteration = iteration + 1;
    
            % 更新参数（Adam优化器）
            [net, averageGrad, averageSqGrad] = adamupdate(net, gradients, averageGrad, averageSqGrad, iteration,current_lr);
        end
    
        % 测试
        test_epoch_iteration = 0;
        test_loss = 0;
        while hasdata(mbq_test)
            [control, state, label] = next(mbq_test);
            current_test_loss = lstm_loss_function(net, state, control, label, L1, L2, L3, state_size, time_step);
            test_loss = test_loss + current_test_loss;
            test_epoch_iteration = test_epoch_iteration + 1;
        end
        test_loss = test_loss/test_epoch_iteration;
    
        % 学习率调度
        if test_loss < best_test_loss
            best_test_loss = test_loss;
            wait_counter = 0;    % 重置计数器
        else
            wait_counter = wait_counter + 1;
            if wait_counter >= patience
                % 降低学习率
                current_lr = max(current_lr * lrReduceFactor, minLearnRate);
                wait_counter = 0; % 重置计数器
                fprintf('学习率降至 %.5f\n', current_lr);
                
                if current_lr == 0
                    break;
                end

            end
        end


        fprintf('Epoch %d, 训练集当前损失: %.4f, 测试集当前损失: %.4f\n', epoch, total_loss, test_loss);
    
        % 保存网络和矩阵
        save([model_savePath, 'trained_network_epoch',num2str(epoch),'.mat'], 'net');  % 保存整个网络
        A = net.Layers(7).Weights;  % 提取矩阵A
        B = net.Layers(8).Weights;  % 提取矩阵B
        save([model_savePath, 'KoopmanMatrix_epoch',num2str(epoch),'.mat'], 'A', 'B');  % 保存A和B矩阵
    
    end
    
    disp('训练完成，网络和矩阵已保存！');



    function [controls, states, labels] = preprocessMiniBatch(controlCell, stateCell, labelCell)
        % 处理control数据（格式转换：CB）
        controls = cat(3, controlCell{:});  % 合并为 [特征数 x batchSize]
        controls = dlarray(controls, 'SSB'); % 转换为dlarray并指定格式
        

        % 合并并重塑state数据
        states = cat(2, stateCell{:});  % 合并为 [特征数 x (batchSize*numTimeSteps)]
        states = dlarray(states, 'CBT');
        
        % 对label执行相同操作
        labels = cat(3, labelCell{:});
        labels = dlarray(labels, 'SSBT');

        % disp(['controlCell的维度是'  num2str(size(controlCell{1}))])
        % disp(['stateCell的维度是'  num2str(size(stateCell{1}))])
        % disp(['labelCell的维度是'  num2str(size(labelCell{1}))])
        % disp(['control的维度是'  num2str(size(controls))])
        % disp(['state的维度是'  num2str(size(states))])
        % disp(['label的维度是'  num2str(size(labels))])
    end
    function [total_loss, gradients] = modelGradients(net, state, control, label, L1, L2, state_size, time_step)
        total_loss = lstm_loss_function(net, state, control, label, L1, L2, L3, state_size, time_step);
        % 计算梯度并梯度裁剪
        gradients = dlgradient(total_loss, net.Learnables);
    end
end