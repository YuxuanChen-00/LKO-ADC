function [best_net, A, B] = train_lko_poly(params, train_data, test_data)
    %% 参数加载
    state_size = params.state_size;
    delay_step = params.delay_step;
    control_size = params.control_size;
    hidden_size = params.hidden_size;
    PhiDimensions = params.PhiDimensions;
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
    control_train = train_data.(fields{1});
    state_train = train_data.(fields{2});
    label_train = train_data.(fields{3});
    state_hd_train = train_data.(fields{4});
    label_hd_train = train_data.(fields{5});

    % 训练集数据存储
    trainControlDatastore = arrayDatastore(control_train, 'IterationDimension', 2);
    trainStateDatastore = arrayDatastore(state_train, 'IterationDimension', 2);
    trainLabelDatastore = arrayDatastore(label_train, 'IterationDimension', 2);
    trainStateHDDatastore = arrayDatastore(state_hd_train, 'IterationDimension', 2);
    trainLabelHDDatastore = arrayDatastore(label_hd_train, 'IterationDimension', 2);
    
    ds_train = combine(trainControlDatastore, trainStateDatastore, trainLabelDatastore, trainStateHDDatastore, trainLabelHDDatastore);
    ds_train = shuffle(ds_train); % 训练集打乱

    %% 网络初始化
    net = lko_poly_network(state_size, control_size, hidden_size, PhiDimensions, delay_step);
    net = net.Net;
    % net = load('models\LKO_POLY_network\trained_network_20250605_2323.mat');
    % net = net.net;

    % fprintf('\n详细层索引列表:\n');
    % for i = 1:numel(net.Layers)
    %     % 显示层索引、层名称和层类型
    %     fprintf('Layer %2d: %-20s (%s)\n',...
    %         i,...
    %         net.Layers(i).Name,...
    %         class(net.Layers(i)));
    % end

 %% 训练设置
    
    averageGrad = [];
    averageSqGrad = [];
    iteration = 0;

    % 初始化学习率调度状态
    best_test_loss = Inf;    % 最佳验证损失
    best_train_loss = Inf;
    wait_counter = 0;        % 无改善计数器
    current_lr = initialLearnRate; % 当前学习率

    %% 主训练循环（移除分阶段逻辑）
    for epoch = 1:num_epochs
        % 重置数据存储
        mbq_train = minibatchqueue(ds_train, ...
            'MiniBatchSize', batchSize, ...
            'MiniBatchFcn', @preprocessMiniBatch, ...
            'OutputEnvironment', device, ...
            'PartialMiniBatch', 'discard');

        % 训练步骤
        while hasdata(mbq_train)
            [control, state, label, state_hd, label_hd] = next(mbq_train);
            % 梯度计算与参数更新
            iteration = iteration + 1;
            [train_loss, gradients] = dlfeval(@modelGradients, net, state, control, label, state_hd, label_hd, L1, L2, L3);
            [net, averageGrad, averageSqGrad] = adamupdate(net, gradients, averageGrad, averageSqGrad, iteration, current_lr);
        end


        % 学习率调度
        if train_loss < best_train_loss
            best_train_loss = train_loss;
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

        if mod(epoch, 100) == 0
            test_loss = zeros(numel(test_data), 1);
            % 测试
            for i = 1:numel(test_data)
                control_test = test_data{i}.control;
                state_test = test_data{i}.state;
                label_test = test_data{i}.label;
                [test_loss(i), ~, ~] = evaluate_lko_poly(net, control_test, state_test, label_test, params.PhiDimensions, params.delay_step);
            end
            if mean(test_loss) < best_test_loss 
                best_test_loss = mean(test_loss);
                % 保存网络和矩阵
                best_net = net;
                A = net.Layers(11).Weights;  % 提取矩阵A
                B = net.Layers(12).Weights;  % 提取矩阵B
            end
        end


        % 日志输出
        fprintf('Epoch %d, 训练损失: %.4f, 测试损失: %.4f, 学习率: %.5f\n',...
                epoch, train_loss, best_test_loss, current_lr);
    end

    % best_net = net;
    % A = net.Layers(11).Weights;  % 提取矩阵A
    % B = net.Layers(12).Weights;  % 提取矩阵B
    
    disp('训练完成，网络和矩阵已保存！');


    function [controls, states, labels, states_hd, labels_hd] = preprocessMiniBatch(controlCell, stateCell, labelCell, statehdCell, labelhdCell)
        % 处理control数据（格式转换：CB）
        controls = cat(2, controlCell{:});  % 合并为 [特征数 x batchSize]
        controls = dlarray(controls, 'CB'); % 转换为dlarray并指定格式
        
        % 合并并重塑state数据
        states = cat(2, stateCell{:});  % 合并为 [特征数 x batchSize]
        states = dlarray(states, 'CB');
        
        % 对label执行相同操作
        labels = cat(2, labelCell{:});
        labels = dlarray(labels, 'CB');

        states_hd = cat(2, statehdCell{:});  % 合并为 [特征数 x batchSize]
        states_hd = dlarray(states_hd, 'CB');

        labels_hd = cat(2, labelhdCell{:});
        labels_hd = dlarray(labels_hd, 'CB');

    end


    function [total_loss, gradients] = modelGradients(net, state, control, label, state_hd, label_hd, L1, L2, L3)
    % 添加梯度阈值参数，默认值为1（如果未提供）
    gradientThreshold = 1.0; % 默认梯度阈值

    
    % 前向传播获取预测值
    total_loss = lko_poly_loss(net, state, control, label, state_hd, label_hd, L1, L2, L3);
    
    % 计算梯度
    gradients = dlgradient(total_loss, net.Learnables);
    
    % 梯度裁剪 (全局L2范数裁剪)
    % gradients = thresholdGlobalL2Norm(gradients, gradientThreshold);
    
    % 梯度裁剪辅助函数
    function clippedGradients = thresholdGlobalL2Norm(gradients, threshold)
        % 计算全局梯度的L2范数
        totalNorm = 0;
        for i = 1:numel(gradients.Value)
            g = gradients.Value{i};
            totalNorm = totalNorm + sum(g.^2, 'all');
        end
        totalNorm = sqrt(extractdata(totalNorm)); % 转换为数值
        
        % 应用裁剪
        if totalNorm > threshold && totalNorm > 0
            scale = threshold / totalNorm;
            clippedGradients = gradients;
            for i = 1:numel(clippedGradients.Value)
                clippedGradients.Value{i} = clippedGradients.Value{i} * scale;
            end
            
            % 显示裁剪信息（调试时使用）
            % fprintf('应用梯度裁剪：%.2f -> %.2f\n', totalNorm, threshold);
        else
            clippedGradients = gradients;
        end
    end
end
end