function data_denormalized = denormalize_data(data_normalized, params)
    % 反归一化函数：根据归一化参数恢复原始数据
    [rows, ~] = size(data_normalized);
    data_denormalized = zeros(size(data_normalized));
        for i = 1:rows
            range_val = params.max_val(i) - params.min_val(i);
            if range_val == 0
                range_val = 1;
            end
            data_denormalized(i, :) = data_normalized(i, :) * range_val + params.min_val(i);
        end

end