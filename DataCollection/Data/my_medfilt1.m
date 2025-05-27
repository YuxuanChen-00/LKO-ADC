function filteredData = my_medfilt1(data, windowSize)
% 自定义一维中值滤波函数
% 输入参数：
%   data - 输入信号矩阵（n×t 二维矩阵）
%   windowSize - 滤波窗口大小（必须为奇数）
% 输出参数：
%   filteredData - 滤波后数据矩阵

% 参数校验
if mod(windowSize, 2) == 0
    error('窗口大小必须为奇数');
end

% 初始化输出矩阵
[n, t] = size(data);
filteredData = zeros(n, t);

% 计算边缘填充量
padSize = (windowSize - 1) / 2;

for row = 1:n
    % 边缘处理：镜像填充（参考网页3的边缘处理方案）
    paddedSignal = [fliplr(data(row,1:padSize)), data(row,:), fliplr(data(row,end-padSize+1:end))];
    
    % 滑动窗口处理
    for i = 1:t
        % 提取当前窗口（参考网页5的排序逻辑）
        window = paddedSignal(i : i+2*padSize);
        
        % 排序取中值（网页2的核心算法）
        sortedWindow = sort(window);
        medianValue = sortedWindow(padSize+1);  % 取中间位置
        
        filteredData(row, i) = medianValue;
    end
end
end