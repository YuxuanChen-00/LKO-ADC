% 文件名：CustomNetwork.m
classdef lko_autoencoder
    properties
        Net             % 存储dlnetwork对象
    end
    
    methods
        function obj = lko_autoencoder(state_size, hidden_size, output_size, control_size, time_step)
            % 构造函数：构建网络结构并初始化
            
            % 基础网络结构
            baseLayers = [
                featureInputLayer(state_size, 'Name', 'state_input')
                fullyConnectedLayer(hidden_size)
                tanhLayer("Name",'tanh1')
                fullyConnectedLayer(hidden_size)
                tanhLayer("Name","tanh2")
            ];
            
            % 创建初始网络
            obj.Net = dlnetwork(baseLayers);

            % 添加额外层
            obj.Net = addLayers(obj.Net, featureInputLayer(control_size, 'Name', 'control_input'));  % 控制输入
            obj.Net = addLayers(obj.Net, fullyConnectedLayer(time_step*state_size + output_size*time_step, 'Name', 'A', 'BiasLearnRateFactor', 0)); % 无偏置线性层A
            obj.Net = addLayers(obj.Net, fullyConnectedLayer(time_step*state_size + output_size*time_step, 'Name', 'B', 'BiasLearnRateFactor', 0)); % 无偏置线性层B
            obj.Net = addLayers(obj.Net, additionLayer(2, 'Name','add')); % 相加A*Phi+B*control_input
            
            recoveryNet = [
                fullyConnectedLayer(hidden_size, 'Name', 'fc_phi2')
                tanhLayer()
                fullyConnectedLayer(hidden_size, 'Name', 'fc_phi3')
                tanhLayer()
                fullyConnectedLayer(state_size, 'Name', 'fc_phi4')    
            ];
            obj.Net = addLayers(obj.Net, recoveryNet);

            % 连接层
            obj.Net = connectLayers(obj.Net, 'tanh2', 'A');
            obj.Net = connectLayers(obj.Net, 'control_input', 'B');
            obj.Net = connectLayers(obj.Net, 'A', 'add/in1');
            obj.Net = connectLayers(obj.Net, 'B', 'add/in2');
            obj.Net = connectLayers(obj.Net, 'add', 'fc_phi2');
            
            analyzeNetwork(obj.Net)

            % 初始化网络
            inputState = dlarray(rand(state_size,8172,time_step),'CBT');
            inputControl = dlarray(rand(control_size,8172),'CB');
            obj.Net = initialize(obj.Net, inputState, inputControl);
            
        end
    end
end