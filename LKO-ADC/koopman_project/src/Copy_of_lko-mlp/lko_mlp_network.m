% 文件名：CustomNetwork.m
classdef lko_mlp_network
    properties
        Net             % 存储dlnetwork对象
    end
    
    methods
        function obj = lko_mlp_network(state_size, control_size, hidden_size, output_size, time_step)

            % 神经网络升维
            encoderLayers = [
                sequenceInputLayer(state_size, 'Name', 'state_input')
                fullyConnectedLayer(hidden_size*4)
                tanhLayer("Name","tanh1")
                fullyConnectedLayer(output_size, 'Name', 'fc_phi')
                tanhLayer('Name','nnOut')
            ];
            obj.Net = addLayers(obj.Net, concatenationLayer(1,2 ,'Name', 'encoder_out'));  
            reshapeLayer0 = functionLayer(@(X) dlarray(reshape(permute(stripdims(X), [1, 3, 2]), ...
                [], size(X, 2)),'CB'), 'Name','toOperator', 'Formattable', true); 
            obj.Net = addLayers(obj.Net, reshapeLayer0);



            % 创建初始网络
            obj.Net = dlnetwork(encoderLayers); 

            % 控制输入层
            obj.Net = addLayers(obj.Net, featureInputLayer(control_size, 'Name', 'control_input'));  
            
            % Koopman线性算子层
            A_layer = fullyConnectedLayer(output_size*time_step, 'Name', 'A', 'BiasLearnRateFactor', 0);
            B_layer = fullyConnectedLayer(output_size*time_step, 'Name', 'B', 'BiasLearnRateFactor', 0);
  

            obj.Net = addLayers(obj.Net, A_layer); % 无偏置线性层A
            obj.Net = addLayers(obj.Net, B_layer); % 无偏置线性层B
            obj.Net = addLayers(obj.Net, additionLayer(2, 'Name','add')); % 相加A*Phi+B*control_input
            % reshapeLayer1 = functionLayer(@(X) dlarray(reshape(stripdims(X), [], size(X, 2), time_step),'CBT'), 'Name', 'pred', 'Formattable', true); 
            % obj.Net = addLayers(obj.Net, reshapeLayer1); 

            % 解码器层
            % decoderLayers = [
            %     sequenceInputLayer(output_size, 'Name', 'decoder_in')
            %     fullyConnectedLayer(state_size*8)
            %     tanhLayer()  
            %     fullyConnectedLayer(state_size*8)
            %     tanhLayer()       
            %     fullyConnectedLayer(state_size*8)
            %     tanhLayer()
            %     fullyConnectedLayer(state_size*4)
            %     tanhLayer()
            %     fullyConnectedLayer(state_size*2)
            %     tanhLayer()
            %     fullyConnectedLayer(state_size)
            %     tanhLayer()
            %     fullyConnectedLayer(state_size, 'Name', 'decoder_out')
            % ];
            % obj.Net = addLayers(obj.Net, decoderLayers);

            % 拼接
            obj.Net = connectLayers(obj.Net, 'state_input', 'encoder_out/in1');
            obj.Net = connectLayers(obj.Net, 'nnOut', 'encoder_out/in2');
            obj.Net = connectLayers(obj.Net, 'encoder_out', 'toOperator');

            % 控制输入和门控输出连接到线性算子层
            obj.Net = connectLayers(obj.Net, 'toOperator', 'A');
            obj.Net = connectLayers(obj.Net, 'control_input', 'B');
            obj.Net = connectLayers(obj.Net, 'A', 'add/in1');
            obj.Net = connectLayers(obj.Net, 'B', 'add/in2');
            obj.Net = connectLayers(obj.Net, 'add', 'pred');


            % 初始化网络
            inputState = dlarray(rand(state_size,1, time_step),'CBT');
            inputControl = dlarray(rand(control_size,1),'CB'); % 此处的 control_size 未乘以 time_step
            % inputDecoder = dlarray(rand(output_size,1, time_step),'CBT');
            obj.Net = initialize(obj.Net, inputState, inputControl);

        end
    end
end