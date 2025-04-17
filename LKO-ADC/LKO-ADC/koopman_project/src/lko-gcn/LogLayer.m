classdef LogLayer < nnet.layer.Layer
    properties
    end
    methods
        function layer = LogLayer(tag)
            layer.Name = tag;
        end
        function X = predict(layer, X)
            disp([layer.Name ' 输入维度: ' mat2str(size(X))]);
            X = X;
        end
    end
end