classdef nn_layer < nnet.layer.Layer
    properties
        % 内部层组件
        FC1
        Tanh1
        FC2
        Tanh2
    end
    
    methods
        function layer = nn_layer(hidden_size, output_size)
            % 构造函数 - 创建内部网络组件
            layer.FC1 = fullyConnectedLayer(hidden_size, 'Name', 'fc1');
            layer.Tanh1 = tanhLayer('Name', 'tanh1');
            layer.FC2 = fullyConnectedLayer(output_size, 'Name', 'fc2');
            layer.Tanh2 = tanhLayer('Name', 'tanh2');
        end
        
        function Z = predict(layer, X)
            % 使用层对象的 predict 方法
            Z = layer.FC1.predict(X);  % 第一层全连接
            Z = layer.Tanh1.predict(Z); % 第一层Tanh
            Z = layer.FC2.predict(Z);  % 第二层全连接
            Z = layer.Tanh2.predict(Z); % 第二层Tanh
        end
        
        function Z = forward(layer, X)
            % 与 predict 相同
            Z = layer.predict(X);
        end
    end
end