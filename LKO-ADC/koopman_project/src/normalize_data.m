function [data_norm, params] = normalize_data(data, params)
    if nargin < 2  % 训练模式：计算并返回参数
        min_val = min(data, [], 2);
        max_val = max(data, [], 2);
        data_norm = (data - min_val) ./ (max_val - min_val);
        params.min_val = min_val;
        params.max_val = max_val;
    else          % 测试模式：直接使用外部参数
        data_norm = (data - params.min_val) ./ (params.max_val - params.min_val);
    end
end