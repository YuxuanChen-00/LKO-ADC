% 文件名：CustomNetwork.m
classdef lko_mlp_network
    properties
        Net             % 存储dlnetwork对象
    end

    methods
        function obj = lko_mlp_network(state_size, control_size ,hidden_size, output_size, time_step)
            % 基础网络结构
            baseLayers = [
                featureInputLayer(state_size*time_step, 'Name','state_input')
                fullyConnectedLayer(hidden_size, 'Name', 'fc_phi')
                reluLayer('Name', 'relu1')
                fullyConnectedLayer(output_size, 'Name', 'fc_phi')
                sigmoidLayer('Name','sigmoid')
            ];
            
            % 创建初始网络
            obj.Net = dlnetwork(baseLayers);
            
            % 添加额外层
            obj.Net = addLayers(obj.Net, featureInputLayer(control_size, 'Name', 'control_input'));  % 控制输入
            obj.Net = addLayers(obj.Net, fullyConnectedLayer(state_size*time_step + output_size, 'Name', 'A', 'BiasLearnRateFactor', 0)); % 无偏置线性层A
            obj.Net = addLayers(obj.Net, fullyConnectedLayer(state_size*time_step + output_size, 'Name', 'B', 'BiasLearnRateFactor', 0)); % 无偏置线性层B
            obj.Net = addLayers(obj.Net, concatenationLayer(1, 2, 'Name', 'concat')); % 拼接state和phi
            obj.Net = addLayers(obj.Net, additionLayer(2, 'Name','add')); % 相加A*Phi+B*control_input


            % 连接层
            obj.Net = connectLayers(obj.Net, 'state_input', 'concat/in1');
            obj.Net = connectLayers(obj.Net, 'sigmoid', 'concat/in2');
            obj.Net = connectLayers(obj.Net, 'concat', 'A');
            obj.Net = connectLayers(obj.Net, 'control_input', 'B');
            obj.Net = connectLayers(obj.Net, 'A', 'add/in1');
            obj.Net = connectLayers(obj.Net, 'B', 'add/in2');

            % 初始化网络
            inputState = dlarray(rand(state_size*time_step,1),'CB');
            inputControl = dlarray(rand(control_size,1),'CB');
            obj.Net = initialize(obj.Net, inputState, inputControl);

        end
    end
end