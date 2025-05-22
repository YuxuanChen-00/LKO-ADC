function [data_norm, params] = normalize_data(data, params)
    if nargin < 2  % 训练模式
        min_val = min(data, [], 2);
        max_val = max(data, [], 2);
        
        % 处理全零行
        zero_rows = (max_val - min_val) == 0;

        % 初始化归一化数据
        data_norm = zeros(size(data));
        
        % 正常行归一化
        valid_rows = ~zero_rows;
        data_norm(valid_rows,:) = (data(valid_rows,:) - min_val(valid_rows)) ./ ...
                                 (max_val(valid_rows) - min_val(valid_rows));
        
        % 全零行保持0（因为原始数据全为0）
        data_norm(zero_rows,:) = 0;
        
        % 存储参数时处理全零行
        params.min_val = min_val;
        params.max_val = max_val;
        params.zero_rows = zero_rows;  % 记录全零行位置
        
    else          % 测试模式
        % 初始化归一化数据
        data_norm = zeros(size(data));
        
        % 正常行处理
        valid_rows = ~params.zero_rows;
        data_norm(valid_rows,:) = (data(valid_rows,:) - params.min_val(valid_rows)) ./ ...
                                (params.max_val(valid_rows) - params.min_val(valid_rows));
        
        % 全零行保持0（根据训练时的参数）
        data_norm(params.zero_rows,:) = 0 - params.min_val(params.zero_rows);  % 保持逻辑一致性
    end
end