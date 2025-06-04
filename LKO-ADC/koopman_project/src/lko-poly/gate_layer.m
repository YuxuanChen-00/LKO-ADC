classdef gate_layer < nnet.layer.Layer
    properties (Learnable)
        % 门控网络权重
        gate_fc1_weights  % 第一层权重 [hidden_dim, (polyDim + nnDim)]
        gate_fc1_bias      % 第一层偏置 [hidden_dim, 1]
        gate_fc2_weights   % 第二层权重 [1, hidden_dim]
        gate_fc2_bias      % 第二层偏置 [1, 1]
    end
    
    properties
        polyDim          % 多项式特征维度
        nnDim            % 神经网络特征维度
        hidden_dim = 16  % 门控网络隐藏层维度
        outDim           % 输出特征维度 (等于max(polyDim, nnDim))
    end
    
    methods
        function layer = gate_layer(polyDim, nnDim, options)
            
            % 解析可选参数
            arguments
                polyDim (1,1) double
                nnDim (1,1) double
                options.hidden_dim = 16
            end
            
            % 设置属性
            layer.polyDim = polyDim;
            layer.nnDim = nnDim;
            layer.outDim = max(polyDim, nnDim);
            layer.hidden_dim = options.hidden_dim;
            
            % 初始化门控网络权重
            layer.gate_fc1_weights = initializeGlorot([layer.hidden_dim, polyDim + nnDim]);
            layer.gate_fc1_bias = zeros(layer.hidden_dim, 1);
            layer.gate_fc2_weights = initializeGlorot([1, layer.hidden_dim]);
            layer.gate_fc2_bias = zeros(1, 1);

            function weights = initializeGlorot(sz)
                % initializeGlorot Glorot权重初始化
                numIn = sz(1);
                numOut = sz(2);
                variance = 2/(numIn + numOut);
                weights = rand(sz) * sqrt(variance) - sqrt(variance)/2;
            end

        end
        
        function Z = predict(layer, polyFeat, nnFeat)
            % predict 前向传播函数
            % 输入:
            %   polyFeat: 多项式特征 [polyDim, batchSize]
            %   nnFeat:   神经网络特征 [nnDim, batchSize]
            %   strain:    应变状态 [3, batchSize] (可选)
            
            % 检查输入维度
            [dimPoly, batchSizePoly] = size(polyFeat);
            [dimNN, batchSizeNN] = size(nnFeat);
            
            assert(dimPoly == layer.polyDim, '多项式特征维度不匹配');
            assert(dimNN == layer.nnDim, '神经网络特征维度不匹配');
            assert(batchSizePoly == batchSizeNN, '批大小不一致');
           
            % 合并特征用于门控计算
            concatFeat = [polyFeat; nnFeat]; % [(polyDim + nnDim), batchSize]
            
            % 计算门控权重 (g_poly)
            g_poly = layer.computeGating(concatFeat);
          
            % 计算神经网络特征权重 (1 - g_poly)
            g_nn = 1 - g_poly;
            
            % 加权相加融合
            Z = g_poly .* polyFeat + g_nn .* nnFeat;
        end
        
        function g_poly = computeGating(layer, concatFeat)
            
            % 第一层: 全连接 + Tanh
            fc1_output = layer.gate_fc1_weights * concatFeat + layer.gate_fc1_bias;
            fc1_activated = tanh(fc1_output);
            
            % 第二层: 全连接 + Sigmoid
            gate_logits = layer.gate_fc2_weights * fc1_activated + layer.gate_fc2_bias;
            g_poly = tanh(gate_logits);
        end
    end
    
end