% 文件名：CustomNetwork.m
classdef lko_poly_network2 
    properties
        Net             % 存储dlnetwork对象
    end
    
    methods
        function obj = lko_poly_network2(state_size, control_size, hidden_size, output_size, time_step)
            input_size = state_size*time_step;
            hidden_size = hidden_size*time_step;
            output_size = output_size*time_step;

            % 神经网络升维
            encoderLayers = [
                featureInputLayer(input_size, 'Name', 'state_input')
                fullyConnectedLayer(hidden_size)
                tanhLayer("Name","tanh1")
                fullyConnectedLayer(output_size, 'Name', 'fc_phi')
                tanhLayer("Name","encoder_out")
            ];

            % 创建初始网络
            obj.Net = dlnetwork(encoderLayers);
            
            % 多项式特征输入层
            obj.Net = addLayers(obj.Net, featureInputLayer(output_size, 'Name', 'poly_input'));  

            % 门控特征融合层 

            % 控制输入层
            obj.Net = addLayers(obj.Net, featureInputLayer(control_size, 'Name', 'control_input'));  
            
            % Koopman线性算子层
            A_layer = fullyConnectedLayer(output_size, 'Name', 'A', 'BiasLearnRateFactor', 0);
            B_layer = fullyConnectedLayer(output_size, 'Name', 'B', 'BiasLearnRateFactor', 0);
            obj.Net = addLayers(obj.Net, A_layer); % 无偏置线性层A
            obj.Net = addLayers(obj.Net, B_layer); % 无偏置线性层B
            obj.Net = addLayers(obj.Net, additionLayer(2, 'Name','add2')); % 相加A*Phi+B*control_input

            % 解码器层
            decoderLayers = [
                featureInputLayer(output_size, 'Name','decoder_in')
                fullyConnectedLayer(hidden_size)
                tanhLayer()
                fullyConnectedLayer(hidden_size)
                tanhLayer()
                fullyConnectedLayer(input_size, 'Name', 'decoder_out')
            ];
            obj.Net = addLayers(obj.Net, decoderLayers);


            % 连接多项式和神经网络特征到融合层
            

            % 控制输入和门控输出连接到线性算子层
            obj.Net = connectLayers(obj.Net, 'add1', 'A');
            obj.Net = connectLayers(obj.Net, 'control_input', 'B');
            obj.Net = connectLayers(obj.Net, 'A', 'add2/in1');
            obj.Net = connectLayers(obj.Net, 'B', 'add2/in2');

            % 初始化网络
            inputState = dlarray(rand(input_size,1),'CB');
            inputPoly = dlarray(rand(output_size,1),'CB');
            inputControl = dlarray(rand(control_size,1),'CB'); % 此处的 control_size 未乘以 time_step
            inputDecoder = dlarray(rand(output_size, 1), 'CB');
            obj.Net = initialize(obj.Net, inputState, inputPoly, inputControl, inputDecoder);

        end
    end
end