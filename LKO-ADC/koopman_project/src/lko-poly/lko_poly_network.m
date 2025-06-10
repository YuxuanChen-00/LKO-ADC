% 文件名：CustomNetwork.m
classdef lko_poly_network 
    properties
        Net             % 存储dlnetwork对象
    end
    
    methods
        function obj = lko_poly_network(state_size, control_size, hidden_size, output_size, time_step)
            input_size = state_size*time_step;
            hidden_size = hidden_size*time_step;
            output_size = output_size*time_step;

            % 神经网络升维
            encoderLayers = [
                featureInputLayer(input_size, 'Name', 'state_input')
                fullyConnectedLayer(hidden_size*4)
                reluLayer("Name","tanh1")
                fullyConnectedLayer(hidden_size*2, 'Name', 'fc_phi')
                reluLayer("Name","tanh1")
                fullyConnectedLayer(hidden_size, 'Name', 'fc_phi')
                reluLayer("Name","encoder_out")
            ];

            % 创建初始网络
            obj.Net = dlnetwork(encoderLayers);
            
            % 多项式特征输入层
            obj.Net = addLayers(obj.Net, featureInputLayer(output_size, 'Name', 'poly_input'));  

            % 特征融合层 
            mixLayers = [
                concatenationLayer(1, 2, 'Name', 'mixLayer')
            ];
            obj.Net = addLayers(obj.Net, mixLayers);

            % 控制输入层
            % obj.Net = addLayers(obj.Net, featureInputLayer(control_size, 'Name', 'control_input'));  
            
            % Koopman线性算子层
            % operator = functionLayer(@(C, X, X_prime) X_prime*pinv([X;C]) , 'Name', 'operator', 'Formattable', true);
            % A_layer = functionLayer(@(X) X(:, 1:output_size), 'Name', 'A_layer', 'Formattable', true);
            % B_layer = functionLayer(@(X) X(:, output_size+1:end), 'Name', 'B_layer', 'Formattable', true);

            % A_layer = fullyConnectedLayer(2*output_size, 'Name', 'A', 'BiasLearnRateFactor', 0);
            % B_layer = fullyConnectedLayer(2*output_size, 'Name', 'B', 'BiasLearnRateFactor', 0);
            % % A_layer.Weights = initial_A;
            % % B_layer.Weights = initial_B;
            % obj.Net = addLayers(obj.Net, operator); % 无偏置线性层B
            % obj.Net = addLayers(obj.Net, A_layer); % 无偏置线性层A
            % obj.Net = addLayers(obj.Net, B_layer); % 无偏置线性层B
            % obj.Net = addLayers(obj.Net, additionLayer(2, 'Name','add2')); % 相加A*Phi+B*control_input
            % obj.Net = connectLayers(obj.Net, 'control_input');
        
            % 解码器层
            % decoderLayers = functionLayer(@(X) X(1:input_size, :), 'Name', 'decoder', 'Formattable', true);
            % obj.Net = addLayers(obj.Net, decoderLayers);

            % 连接解码器和线性算子层
            % obj.Net = connectLayers(obj.Net, 'mixLayer', 'decoder');

            % 连接多项式和神经网络特征到门控融合层
            obj.Net = connectLayers(obj.Net, 'poly_input', 'mixLayer/in1');
            obj.Net = connectLayers(obj.Net, 'encoder_out', 'mixLayer/in2');

            % 控制输入和门控输出连接到线性算子层
            % obj.Net = connectLayers(obj.Net, 'mixLayer', 'A');
            % obj.Net = connectLayers(obj.Net, 'control_input', 'B');
            % obj.Net = connectLayers(obj.Net, 'A', 'add2/in1');
            % obj.Net = connectLayers(obj.Net, 'B', 'add2/in2');


            % 初始化网络
            inputState = dlarray(rand(input_size,1),'CB');
            inputPoly = dlarray(rand(output_size,1),'CB');
            % inputControl = dlarray(rand(control_size,1),'CB'); % 此处的 control_size 未乘以 time_step
            obj.Net = initialize(obj.Net, inputState, inputPoly);

        end
    end
end