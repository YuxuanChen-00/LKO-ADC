function [net, A, B] = train_mlp_lko(params, train_data, test_data, model_savePath)
    %% 参数加载
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
    control_train = train_data.(fields{1});
    state_train = train_data.(fields{2});
    label_train = train_data.(fields{3});
    
    control_test = test_data.(fields{1});
    state_test = test_data.(fields{2});
    label_test = test_data.(fields{3});
    
    disp(size(label_train))

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
    net = lko_mlp_network(state_size, control_size ,hidden_size, output_size, time_step);
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
    
    averageGrad = [];
    averageSqGrad = [];
    iteration = 0;

    % 初始化学习率调度状态
    best_test_loss = Inf;    % 最佳验证损失
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
        mbq_test = minibatchqueue(ds_test, ...
            'MiniBatchSize', batchSize, ...
            'MiniBatchFcn', @preprocessMiniBatch, ...
            'OutputEnvironment', device, ...
            'PartialMiniBatch', 'discard');

        % 训练步骤
        while hasdata(mbq_train)
            [control, state, label] = next(mbq_train);
            % 梯度计算与参数更新
            iteration = iteration + 1;
            [total_loss, gradients] = dlfeval(@modelGradients, net, state, control, label, L1, L2, L3, state_size, time_step);
            [net, averageGrad, averageSqGrad] = adamupdate(net, gradients, averageGrad, averageSqGrad, iteration, current_lr);
        end

        % 测试步骤
        test_epoch_iteration = 0;
        test_loss = 0;
        while hasdata(mbq_test)
            [control, state, label] = next(mbq_test);
            current_test_loss = mlp_loss_function(net, state, control, label, L1, L2, L3, state_size, time_step);
            test_loss = test_loss + current_test_loss;
            test_epoch_iteration = test_epoch_iteration + 1;
        end
        test_loss = test_loss / test_epoch_iteration;

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


        % 日志输出
        fprintf('Epoch %d, 训练损失: %.4f, 测试损失: %.4f, 学习率: %.5f\n',...
                epoch, total_loss, test_loss, current_lr);

        % 模型保存
        if mod(epoch, 100) == 0
            save([model_savePath, 'mlp_network_epoch',num2str(epoch),'.mat'], 'net');
            A = net.Layers(7).Weights;
            B = net.Layers(8).Weights;
            save([model_savePath, 'mlp_KoopmanMatrix_epoch',num2str(epoch),'.mat'], 'A', 'B');
        end
    end
    
    disp('训练完成，网络和矩阵已保存！');


    function [controls, states, labels] = preprocessMiniBatch(controlCell, stateCell, labelCell)
        % 处理control数据（格式转换：CB）
        controls = cat(3, controlCell{:});  % 合并为 [特征数 x batchSize]
        controls = dlarray(controls, 'SCB'); % 转换为dlarray并指定格式
        
        % 处理state和label数据（格式转换：CBT）
        % 获取维度信息
    
        % 合并并重塑state数据
        states = cat(2, stateCell{:});  % 合并为 [特征数 x batchSize]
        states = dlarray(states, 'CB');
        
        % 对label执行相同操作
        labels = cat(3, labelCell{:});
        labels = dlarray(labels, 'SCB');
        % 
        % disp(['controlCell的维度是'  num2str(size(controlCell{1}))])
        % disp(['stateCell的维度是'  num2str(size(stateCell{1}))])
        % disp(['labelCell的维度是'  num2str(size(labelCell{1}))])
        % disp(['control的维度是'  num2str(size(controls))])
        % disp(['state的维度是'  num2str(size(states))])
        % disp(['label的维度是'  num2str(size(labels))])

    end


    function [total_loss, gradients] = modelGradients(net, state, control, label, L1, L2,L3, state_size, time_step)
        % 前向传播获取预测值
        total_loss = mlp_loss_function(net, state, control, label, L1, L2, L3, state_size, time_step);
        % 计算梯度并梯度裁剪
        gradients = dlgradient(total_loss, net.Learnables);
    end
end