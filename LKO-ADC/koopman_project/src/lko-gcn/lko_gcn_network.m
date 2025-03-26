% 文件名：CustomNetwork.m
classdef lko_gcn_network
    properties
        Net             % 存储dlnetwork对象
    end

    methods
        function obj = lko_gcn_network(feature_size, node_size, adjMatrix, hidden_size, output_size, control_size)
            % 基础网络结构
            baseLayers = [
                imageInputLayer([feature_size, node_size, 1], 'Name','state_input')
                GraphConvolutionLayer(feature_size, hidden_size, adjMatrix, 'graph')
                reluLayer('Name', 'relu')
                functionLayer(@(X) dlarray(reshape(stripdims(X), [], size(X, 3)),'CB'), 'Name', 'reshape1', 'Formattable', true) 
                fullyConnectedLayer(output_size, 'Name', 'fc_phi')
            ];
            
            % 创建初始网络
            obj.Net = dlnetwork(baseLayers);
            
            % 添加额外层
            obj.Net = addLayers(obj.Net, featureInputLayer(control_size, 'Name', 'control_input'));  % 控制输入
            obj.Net = addLayers(obj.Net, fullyConnectedLayer(feature_size*node_size + output_size, 'Name', 'A', 'BiasLearnRateFactor', 0)); % 无偏置线性层A
            obj.Net = addLayers(obj.Net, fullyConnectedLayer(feature_size*node_size + output_size, 'Name', 'B', 'BiasLearnRateFactor', 0)); % 无偏置线性层B
            obj.Net = addLayers(obj.Net, concatenationLayer(1, 2, 'Name', 'concat')); % 拼接state和phi
            obj.Net = addLayers(obj.Net, additionLayer(2, 'Name','add')); % 相加A*Phi+B*control_input
            obj.Net = addLayers(obj.Net, functionLayer(@(X) dlarray(reshape(stripdims(X), [], size(X, 4)),'CB'), 'Name', 'reshape', 'Formattable', true));



            % 连接层
            obj.Net = connectLayers(obj.Net, 'state_input', 'reshape');
            obj.Net = connectLayers(obj.Net, 'reshape', 'concat/in1');
            obj.Net = connectLayers(obj.Net, 'fc_phi', 'concat/in2');
            obj.Net = connectLayers(obj.Net, 'concat', 'A');
            obj.Net = connectLayers(obj.Net, 'control_input', 'B');
            obj.Net = connectLayers(obj.Net, 'A', 'add/in1');
            obj.Net = connectLayers(obj.Net, 'B', 'add/in2');

            % 初始化网络
            inputState = dlarray(rand(feature_size,node_size,1,1),'SSCB');
            inputControl = dlarray(rand(control_size,1),'CB');
            obj.Net = initialize(obj.Net, inputState, inputControl);

        end
    end
end