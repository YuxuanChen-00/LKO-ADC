function trajectory = generateReferenceSquare(center_pos, dist_center_to_vertex, total_steps)
    % 生成以z轴为法向量的正方形轨迹
    % 输入:
    %   center_pos: [xc, yc, zc] 正方形中心点位置 (基坐标系)
    %   dist_center_to_vertex: 中心点到顶点的长度
    %   total_steps: 轨迹上的总点数 (建议 >= 4)
    % 输出:
    %   trajectory: 3xN矩阵，每列包括位置[x; y; z]
    
    % --- 参数检查 ---
    if total_steps < 4
        warning('总点数(total_steps)小于4，轨迹可能无法表示完整的正方形。');
    end
    
    % --- 坐标与尺寸计算 ---
    % 从输入参数中提取中心坐标
    xc = center_pos(1);
    yc = center_pos(2);
    zc = center_pos(3);
    
    % 计算正方形的半边长、边长和总周长
    half_side_length = dist_center_to_vertex / sqrt(2);
    L = half_side_length;
    side_length = 2 * L;
    perimeter = 4 * side_length;
    
    % --- 定义轨迹路径的关键点 ---
    % 定义四个顶点 (以右下角为起点，逆时针顺序)
    v1 = [xc + L, yc - L];
    v2 = [xc + L, yc + L];
    v3 = [xc - L, yc + L];
    v4 = [xc - L, yc - L];
    
    % 创建一个闭环的路径，包含5个点 (起点、三个顶点、回到起点)
    path_waypoints_x = [v1(1), v2(1), v3(1), v4(1), v1(1)];
    path_waypoints_y = [v1(2), v2(2), v3(2), v4(2), v1(2)];
    
    % 定义路径上每个关键点对应的累计距离
    path_distances = [0, side_length, 2 * side_length, 3 * side_length, perimeter];
    
    % --- 均匀插值生成轨迹点 ---
    % 在总周长上生成 'total_steps' 个均匀分布的采样距离
    % 采用 (0:N-1)/N * P 的方式，可以使生成的点均匀分布，且终点不会与起点重合
    % 这对于连续循环的轨迹跟踪很有用
    sample_distances = (0 : total_steps - 1) * (perimeter / total_steps);
    
    % 使用线性插值(interp1)来计算每个采样距离上的x, y坐标
    x = interp1(path_distances, path_waypoints_x, sample_distances);
    y = interp1(path_distances, path_waypoints_y, sample_distances);
    
    % z坐标保持不变
    z = zc * ones(size(x));
    
    % --- 合并轨迹数据 ---
    trajectory = [x; y; z];

end