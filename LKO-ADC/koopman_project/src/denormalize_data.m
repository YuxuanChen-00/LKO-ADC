function data_denormalized = denormalize_data(data_normalized, params)
% DENORMALIZE_DATA_ZSCORE 将Z-score标准化后的数据恢复为原始尺度
%   输入:
%       data_normalized - 标准化后的数据矩阵 (每行是一个特征)
%       params          - 包含标准化参数的结构体 (需有mu和sigma字段)
%   输出:
%       data_denormalized - 反标准化后的原始数据

% 初始化输出矩阵
data_denormalized = zeros(size(data_normalized));

% 获取特征数量(行数)
[rows, ~] = size(data_normalized);

% 对每个特征执行反标准化
for i = 1:rows
    % Z-score反标准化公式: x = z * σ + μ
    data_denormalized(i, :) = data_normalized(i, :) * params.sigma(i) + params.mu(i);
end
end