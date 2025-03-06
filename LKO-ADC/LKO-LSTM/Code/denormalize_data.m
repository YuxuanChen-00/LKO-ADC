function data_denormalized = denormalize_data(data_normalized, params)
    % 反归一化函数：根据归一化参数恢复原始数据
    %
    % 输入：
    %   data_normalized : 归一化后的数据（12×t）
    %   params          : 归一化参数结构体
    %
    % 输出：
    %   data_denormalized : 反归一化后的原始数据（12×t）

    [rows, ~] = size(data_normalized);
    data_denormalized = zeros(size(data_normalized));
    
    switch lower(params.method)
        case 'minmax'
            for i = 1:rows
                range_val = params.max_vals(i) - params.min_vals(i);
                if range_val == 0
                    range_val = 1;
                end
                data_denormalized(i, :) = data_normalized(i, :) * range_val + params.min_vals(i);
            end
            
        case 'zscore'
            for i = 1:rows
                data_denormalized(i, :) = data_normalized(i, :) * params.sigma(i) + params.mu(i);
            end
            
        case 'range'
            a = params.target_range(1);
            b = params.target_range(2);
            for i = 1:rows
                range_val = params.max_vals(i) - params.min_vals(i);
                if range_val == 0
                    range_val = 1;
                end
                data_denormalized(i, :) = params.min_vals(i) + (data_normalized(i, :) - a) * range_val / (b - a);
            end
            
        case 'custom'
            range_val = params.global_max - params.global_min;
            if range_val == 0
                range_val = 1;
            end
            data_denormalized = data_normalized * range_val + params.global_min;
            
        otherwise
            error('未知的归一化方法');
    end
end