function [data_normalized, params] = normalize_data(data, method, varargin)
    % 归一化函数：按行（特征）归一化数据
    %
    % 输入：
    %   data   : 原始数据矩阵（12×t）
    %   method : 归一化方法，可选 'minmax'（默认）, 'zscore', 'range', 'custom'
    %   varargin: 自定义参数（例如范围 [a,b]）
    %
    % 输出：
    %   data_normalized : 归一化后的数据（12×t）
    %   params          : 结构体，保存归一化参数（方法、min/max、mu/sigma等）

    % 默认参数
    if nargin < 2
        method = 'minmax';
    end
    
    [rows, ~] = size(data);
    params = struct();
    params.method = method;
    data_normalized = zeros(size(data));
    
    switch lower(method)
        case 'minmax'  % 最大-最小值归一化到 [0,1]
            params.min_vals = zeros(rows, 1);
            params.max_vals = zeros(rows, 1);
            for i = 1:rows
                params.min_vals(i) = min(data(i, :));
                params.max_vals(i) = max(data(i, :));
                range_val = params.max_vals(i) - params.min_vals(i);
                if range_val == 0
                    range_val = 1;  % 避免除零
                end
                data_normalized(i, :) = (data(i, :) - params.min_vals(i)) / range_val;
            end
            
        case 'zscore'  % Z-score标准化（均值为0，标准差为1）
            params.mu = zeros(rows, 1);
            params.sigma = zeros(rows, 1);
            for i = 1:rows
                params.mu(i) = mean(data(i, :));
                params.sigma(i) = std(data(i, :));
                if params.sigma(i) == 0
                    params.sigma(i) = 1;  % 避免除零
                end
                data_normalized(i, :) = (data(i, :) - params.mu(i)) / params.sigma(i);
            end
            
        case 'range'   % 缩放到指定范围（例如 [-1,1]）
            if nargin < 3
                target_range = [-1, 1];  % 默认范围
            else
                target_range = varargin{1};
            end
            a = target_range(1);
            b = target_range(2);
            params.min_vals = zeros(rows, 1);
            params.max_vals = zeros(rows, 1);
            for i = 1:rows
                params.min_vals(i) = min(data(i, :));
                params.max_vals(i) = max(data(i, :));
                range_val = params.max_vals(i) - params.min_vals(i);
                if range_val == 0
                    range_val = 1;
                end
                data_normalized(i, :) = a + (data(i, :) - params.min_vals(i)) * (b - a) / range_val;
            end
            
        case 'custom'  % 自定义缩放（例如已知全局范围）
            if nargin < 3
                error('需提供全局最小值和最大值');
            end
            global_min = varargin{1};
            global_max = varargin{2};
            params.global_min = global_min;
            params.global_max = global_max;
            range_val = global_max - global_min;
            if range_val == 0
                range_val = 1;
            end
            data_normalized = (data - global_min) / range_val;
            
        otherwise
            error('不支持的归一化方法');
    end
end