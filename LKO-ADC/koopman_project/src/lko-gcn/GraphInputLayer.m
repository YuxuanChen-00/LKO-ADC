classdef GraphInputLayer < nnet.layer.Layer
    properties
        F   % 特征维度
        N   % 节点数
    end
    
    methods
        function layer = GraphInputLayer(F, N, name)
            % 构造函数：必须指定 F 和 N
            % 参数：
            %   F: 特征维度（整数）
            %   N: 节点数（整数）
            %   name: 层名称（默认 'graph_input'）
            
            arguments
                F (1,1) {mustBePositiveInteger}
                N (1,1) {mustBePositiveInteger}
                name = 'graph_input'
            end
            
            % 初始化父类
            layer = layer@nnet.layer.Layer();
            layer.Name = name;
            
            % 保存维度
            layer.F = F;
            layer.N = N;
        end
        
        function X = predict(layer, X)
            % 前向传播：验证输入维度
            [input_F, input_N, input_B] = size(X);
            if input_F ~= layer.F || input_N ~= layer.N
                error('输入特征维度必须为 F×N×B（F=%d, N=%d），实际维度为 %d×%d×%d', ...
                    layer.F, layer.N, input_F, input_N, input_B);
            end
        end
    end
end

% 辅助函数：验证正整数
function mustBePositiveInteger(a)
    if ~(isscalar(a) && isnumeric(a) && a > 0 && floor(a) == a)
        error('输入必须为正整数');
    end
end