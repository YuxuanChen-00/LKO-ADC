classdef gate_layer < nnet.layer.Layer & nnet.layer.Formattable
    properties (Learnable)
        % 门控网络权重
        gate_fc1_weights  % 第一层权重 [hidden_dim, (2*input_dim)]
        gate_fc1_bias      % 第一层偏置 [hidden_dim, 1]
        gate_fc2_weights   % 第二层权重 [1, hidden_dim]
        gate_fc2_bias      % 第二层偏置 [1, 1]
    end
    
    properties
        input_dim
        hidden_dim = 16  % 门控网络隐藏层维度
        g_poly = 0.5
    end
    
    methods
        function layer = gate_layer(input_dim, hidden_dim, varargin)
            % 解析名称-值对参数
            p = inputParser;
            addParameter(p, 'Name', '');
            parse(p, varargin{:});
            
            layer.Name = p.Results.Name; % 设置层名称
            layer.input_dim = input_dim;
            layer.hidden_dim = hidden_dim;
            layer.InputNames = {'polyFeat', 'nnFeat'};

            % 初始化门控网络权重
            % For WX+b, W is [num_neurons_current_layer, num_neurons_previous_layer]
            % Input to first FC is 2*input_dim features, output is hidden_dim features
            layer.gate_fc1_weights = gate_layer.initializeGlorot_static([layer.hidden_dim, 2*layer.input_dim]);
            layer.gate_fc1_bias = zeros(layer.hidden_dim, 1); % MATLAB will auto-convert to dlarray if needed

            % Input to second FC is hidden_dim features, output is 1 feature
            layer.gate_fc2_weights = gate_layer.initializeGlorot_static([1, layer.hidden_dim]);
            layer.gate_fc2_bias = zeros(1, 1);
        end
        
        function Z = predict(layer, polyFeat, nnFeat)
            % predict 前向传播函数
            % polyFeat and nnFeat are expected to be dlarrays, likely with 'CB' format 
            % e.g., size [feature_dimension, batch_size]

            % 检查输入维度
            % Assuming polyFeat and nnFeat have features along the first dimension
            assert(size(polyFeat,1) == layer.input_dim, ...
                ['polyFeat has ', num2str(size(polyFeat,1)), ' features, but layer.input_dim is ', num2str(layer.input_dim)]);
            assert(size(nnFeat,1) == layer.input_dim, ...
                ['nnFeat has ', num2str(size(nnFeat,1)), ' features, but layer.input_dim is ', num2str(layer.input_dim)]);
            assert(size(polyFeat,2) == size(nnFeat,2), 'Batch sizes of polyFeat and nnFeat must be consistent');
          
            % 合并特征用于门控计算
            concatFeat = [polyFeat; nnFeat]; % Concatenates along the first dimension (features)
                                            % Size will be [2*layer.input_dim, batchSize]
            
            % 计算门控权重 (g_poly)
            layer.g_poly = layer.computeGating(concatFeat);
          
            % 计算神经网络特征权重 (1 - g_poly)
            g_nn = 1 - layer.g_poly;
            
            % 加权相加融合
            Z = layer.g_poly .* polyFeat + g_nn .* nnFeat;
        end
        
        function g_poly = computeGating(layer, concatFeat)
            % concatFeat is [2*layer.input_dim, batchSize]

            % 第一层: 全连接 + Tanh
            % Use fullyconnect for dlarray operations
            fc1_output = fullyconnect(concatFeat, layer.gate_fc1_weights, layer.gate_fc1_bias);
            fc1_activated = tanh(fc1_output); % fc1_activated is [hidden_dim, batchSize]
            
            % 第二层: 全连接 + Tanh (as per original, consider sigmoid if gate needs to be in [0,1])
            % Use fullyconnect for dlarray operations
            gate_logits = fullyconnect(fc1_activated, layer.gate_fc2_weights, layer.gate_fc2_bias);
            g_poly = tanh(gate_logits); % g_poly is [1, batchSize]
        end
    end

    methods (Static)
        function weights = initializeGlorot_static(sz)
            % initializeGlorot Glorot权重初始化
            % sz should be [number_of_outputs, number_of_inputs] for the layer
            fan_out = sz(1); % Number of output neurons for this FC layer
            fan_in = sz(2);  % Number of input features to this FC layer
            
            variance = 2/(fan_in + fan_out);
            % rand(sz) produces values in [0,1]
            % weights will be in [-sqrt(variance)/2, sqrt(variance)/2] with mean 0
            weights = rand(sz) * sqrt(variance) - sqrt(variance)/2;
        end
    end
end