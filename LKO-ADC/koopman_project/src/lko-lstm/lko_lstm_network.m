% 文件名：CustomNetwork.m
classdef lko_lstm_network 
    properties
        Net             % 存储dlnetwork对象
    end
    
    methods
        function obj = lko_lstm_network(state_size, hidden_size, output_size, control_size, time_step)
            % 构造函数：构建网络结构并初始化
            
            % 基础网络结构
            baseLayers = [
                sequenceInputLayer(state_size, 'Name', 'state_input')
                % lstmLayer(hidden_size, 'OutputMode', 'sequence', 'Name', 'lstm1')
                % reluLayer("Name",'relu')
                % lstmLayer(hidden_size, 'OutputMode', 'sequence', 'Name', 'lstm2')
                fullyConnectedLayer(hidden_size)
                tanhLayer("Name","relu1")
                fullyConnectedLayer(hidden_size*2)
                fullyConnectedLayer(output_size, 'Name', 'fc_phi')
                reluLayer("Name","relu")
            ];
            
            % 创建初始网络
            obj.Net = dlnetwork(baseLayers);
            
            % 添加额外层
            obj.Net = addLayers(obj.Net, featureInputLayer(control_size, 'Name', 'control_input'));  % 控制输入
            obj.Net = addLayers(obj.Net, fullyConnectedLayer(time_step*state_size + output_size*time_step, 'Name', 'A', 'BiasLearnRateFactor', 0)); % 无偏置线性层A
            obj.Net = addLayers(obj.Net, fullyConnectedLayer(time_step*state_size + output_size*time_step, 'Name', 'B', 'BiasLearnRateFactor', 0)); % 无偏置线性层B
            obj.Net = addLayers(obj.Net, concatenationLayer(1, 2, 'Name', 'concat')); % 拼接state和phi
            obj.Net = addLayers(obj.Net, additionLayer(2, 'Name','add')); % 相加A*Phi+B*control_input
            obj.Net = addLayers(obj.Net, functionLayer(@(X) dlarray(reshape(permute(stripdims(X), [1, 3, 2]), [], size(X, 2)),'CB'), 'Name', 'reshape1', 'Formattable', true)); % 三维转二维数据
            obj.Net = addLayers(obj.Net, functionLayer(@(X) dlarray(reshape(permute(stripdims(X), [1, 3, 2]), [], size(X, 2)),'CB'), 'Name', 'reshape2', 'Formattable', true)); % 三维转二维数据
            
            % obj.Net = addLayers(obj.Net, flattenLayer('Name','reshape1'));
            % obj.Net = addLayers(obj.Net, flattenLayer('Name','reshape2'));

            % 连接层
            obj.Net = connectLayers(obj.Net, 'state_input', 'reshape1');
            obj.Net = connectLayers(obj.Net, 'relu', 'reshape2');
            obj.Net = connectLayers(obj.Net, 'reshape1', 'concat/in1');
            obj.Net = connectLayers(obj.Net, 'reshape2', 'concat/in2');
            obj.Net = connectLayers(obj.Net, 'concat', 'A');
            obj.Net = connectLayers(obj.Net, 'control_input', 'B');
            obj.Net = connectLayers(obj.Net, 'A', 'add/in1');
            obj.Net = connectLayers(obj.Net, 'B', 'add/in2');

            % 初始化网络
            inputState = dlarray(rand(state_size,8172,time_step),'CBT');
            inputControl = dlarray(rand(control_size,8172),'CB');
            obj.Net = initialize(obj.Net, inputState, inputControl);

        end
    end
end