% 文件名：CustomNetwork.m
classdef ae_network
    properties
        Net             % 存储dlnetwork对象
    end

    methods
        function obj = lko_gcn_network(state_size, control_size ,hidden_size, output_size)
            % 基础网络结构
            baseLayers = [
                featureInputLayer(state_size, 'Name','state_input')
                fullyConnectedLayer(hidden_size, 'Name', 'fc_phi')
                reluLayer('Name', 'relu1')
                fullyConnectedLayer(output_size, 'Name', 'fc_phi')
                tanhLayer('Name','tanh')
            ];
            
            % 创建初始网络
            obj.Net = dlnetwork(baseLayers);
            % 初始化网络
            inputState = dlarray(rand(input_size,1),'CB');
            inputControl = dlarray(rand(control_size,1),'CB');
            obj.Net = initialize(obj.Net, inputState, inputControl);

        end
    end
end