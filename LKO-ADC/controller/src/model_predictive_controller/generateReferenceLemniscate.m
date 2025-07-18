function trajectory = generateReferenceLemniscate(center_pos, max_distance, total_steps)
    % 生成以z轴为法向量的无穷形（倒8字）轨迹
    % 该轨迹基于伯努利双纽线 (Lemniscate of Bernoulli) 的参数方程
    % 输入:
    %   center_pos:     [xc, yc, zc] 轨迹中心点位置 (基坐标系)
    %   max_distance:   中心到最远点的距离 (即双纽线参数 'a')
    %   total_steps:    轨迹上的总点数
    % 输出:
    %   trajectory: 3xN矩阵，每列包括位置[x; y; z]
    
    % --- 参数检查 ---
    if total_steps <= 0
        error('总点数(total_steps)必须为正数。');
    end
    
    % --- 坐标与参数提取 ---
    xc = center_pos(1);
    yc = center_pos(2);
    zc = center_pos(3);
    a = max_distance; % 'a' 在双纽线方程中代表中心到最远点的距离
    
    % --- 生成参数变量 t ---
    % 在 [0, 2*pi) 区间内生成 'total_steps' 个均匀分布的角度
    % 使用 (0:N-1)/N * 2*pi 的方式可确保轨迹闭合但首尾点不重合
    t = (0 : total_steps - 1) * (2 * pi / total_steps);
    
    % --- 应用伯努利双纽线参数方程 ---
    % x = a * cos(t) / (1 + sin(t)^2)
    % y = a * sin(t) * cos(t) / (1 + sin(t)^2)
    % 注意 MATLAB 中的点运算 (./, .*, .^) 用于向量的逐元素计算
    denominator = 1 + sin(t).^2;
    x = xc + a * cos(t) ./ denominator;
    y = yc + a * sin(t) .* cos(t) ./ denominator;
    
    % z坐标保持不变
    z = zc * ones(size(x));
    
    % --- 合并轨迹数据 ---
    % 将x, y, z坐标合并为最终的 3xN 轨迹矩阵
    trajectory = [x; y; z];

end