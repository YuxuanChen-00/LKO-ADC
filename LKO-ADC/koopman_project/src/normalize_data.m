function [data_norm, params] = normalize_data(data, params)
% NORMALIZE_DATA 使用Z-score方法对数据进行标准化
%   [data_norm, params] = normalize_data(data)
%   在训练模式下调用，计算并返回标准化后的数据和统计参数（均值、标准差）
%
%   data_norm = normalize_data(data, params)
%   在测试模式下调用，使用已有的统计参数来标准化新数据

if nargin < 2  % 训练模式: 计算并存储参数
    % 1. 计算每一行（每一个特征）的均值和标准差
    mu = mean(data, 2);
    sigma = std(data, 0, 2); % 沿第2维（行）计算标准差，使用n-1无偏估计

    % 2. 找到标准差为零的行（即该行所有元素都相同）
    %    为避免除以零，需要对这些行进行特殊处理
    zero_std_rows = (sigma == 0);

    % 初始化标准化数据矩阵
    data_norm = zeros(size(data));

    % 3. 对标准差不为零的"正常"行进行Z-score标准化
    valid_rows = ~zero_std_rows;
    if any(valid_rows)
        data_norm(valid_rows,:) = (data(valid_rows,:) - mu(valid_rows)) ./ sigma(valid_rows);
    end

    % 4. 标准差为零的行，标准化后为全零
    %    因为 (x - mu) 对于这些行本身就是0
    if any(zero_std_rows)
        data_norm(zero_std_rows,:) = 0;
    end
    
    % 5. 存储参数以备后续使用
    params.mu = mu;
    params.sigma = sigma;
    params.zero_std_rows = zero_std_rows;

else          % 测试模式: 使用已有的参数
    % 初始化标准化数据矩阵
    data_norm = zeros(size(data));
    
    % 1. 获取需要正常处理的行
    valid_rows = ~params.zero_std_rows;
    
    % 2. 使用训练时得到的均值和标准差进行标准化
    if any(valid_rows)
        data_norm(valid_rows,:) = (data(valid_rows,:) - params.mu(valid_rows)) ./ params.sigma(valid_rows);
    end

    % 3. 标准差为零的行同样处理为全零
    if any(params.zero_std_rows)
        data_norm(params.zero_std_rows,:) = 0;
    end
end
end