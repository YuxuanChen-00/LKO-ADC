classdef GraphConvolutionLayer < nnet.layer.Layer
    properties (Learnable)
        Weights   % 可学习的权重矩阵
    end
    
    properties
        AdjMatrix % 固定的邻接矩阵（定义在 GCN 层中）
    end
    
    methods
        function layer = GraphConvolutionLayer(inputSize, outputSize, adjMatrix, name)
            % 构造函数：必须指定 inputSize, outputSize, adjMatrix
            % 参数：
            %   inputSize: 输入特征维度（整数）
            %   outputSize: 输出特征维度（整数）
            %   adjMatrix: 邻接矩阵（N×N）
            %   name: 层名称（默认 'gcn'）
            
            arguments
                inputSize (1,1) {mustBePositiveInteger}
                outputSize (1,1) {mustBePositiveInteger}
                adjMatrix (:,:) {mustBeSquareMatrix}
                name = 'gcn'
            end
            
            % 初始化父类
            layer = layer@nnet.layer.Layer();
            layer.Name = name;
            
            % 保存邻接矩阵并验证维度
            layer.AdjMatrix = adjMatrix;
            
            % 初始化可学习权重
            layer.Weights = randn(inputSize, outputSize) * 0.01;
        end
        
        function Z = predict(layer, X)
            % 前向传播：执行图卷积
            % 输入 X 的维度应为 [inputSize × N × B]
            % 输出 Z 的维度为 [outputSize × N × B]
            
            X = squeeze(X);
            

            % 提取参数
            W = layer.Weights;
            A = layer.AdjMatrix;
            
            % 验证输入维度
            [~, N_input, ~] = size(X);
            [N_adj, ~] = size(A);
            if N_input ~= N_adj
                error('输入节点数 (%d) 与邻接矩阵维度 (%d×%d) 不匹配', N_input, N_adj, N_adj);
            end
            
            % 图卷积计算
            Z = pagemtimes(W', X);  % 线性变换: [outputSize × N × B]
            Z = pagemtimes(Z, A);   % 邻接矩阵聚合: [outputSize × N × B]
        end
    end
end

% 辅助函数：验证邻接矩阵是方阵
function mustBeSquareMatrix(a)
    if ~isequal(size(a,1), size(a,2))
        error('邻接矩阵必须是方阵');
    end
end

% 辅助函数：验证正整数
function mustBePositiveInteger(a)
    if ~(isscalar(a) && isnumeric(a) && a > 0 && floor(a) == a)
        error('输入必须为正整数');
    end
end