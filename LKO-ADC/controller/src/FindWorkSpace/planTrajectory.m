function [circle_center, radius, start_point] = planTrajectory(workspace_points)
% planCircleInWorkspace: 在给定的三维点云工作空间内规划一个最优的圆形轨迹。
%
% 输入参数:
%   workspace_points: 一个 3xM 的矩阵，代表机械臂工作空间的所有采样点。
%
% 输出:
%   circle_center:  一个 3x1 的向量，代表规划出的圆的圆心 [xc; yc; zc]。
%   radius:         一个标量，代表规划出的圆的半径。
%   start_point:    一个 3x1 的向量，代表圆上一个合适的起始点。

% --- 步骤 0: 输入检查 ---
if size(workspace_points, 1) ~= 3 || size(workspace_points, 2) < 10
    error('输入的工作空间点云必须是一个 3xM 的矩阵，且至少包含10个点。');
end

fprintf('开始在工作空间中规划圆形轨迹...\n');

% --- 步骤 1 & 2: 计算3D凸包并确定最佳Z平面 ---
fprintf('  > 步骤1: 计算三维凸包以确定最佳Z平面...\n');
try
    % 计算三维凸包的顶点索引
    k_3d = convhull(workspace_points(1,:)', workspace_points(2,:)', workspace_points(3,:)');
    % 获取所有顶点的索引
    hull_vertices_indices = unique(k_3d(:));
    % 获取所有顶点的坐标
    hull_vertices_3d = workspace_points(:, hull_vertices_indices);
    % 将3D凸包的质心作为最佳位置
    centroid_3d = mean(hull_vertices_3d, 2);
    z_optimal = centroid_3d(3);
catch ME
    warning('无法计算三维凸包，可能因为点共面。将使用所有点的Z坐标平均值。');
    z_optimal = mean(workspace_points(3,:));
end
fprintf('  > 最佳Z平面位置确定为: %.3f\n', z_optimal);


% --- 步骤 3: 投影到2D并计算2D凸包 ---
fprintf('  > 步骤2: 将工作空间投影到XY平面并计算其二维轮廓...\n');
workspace_points_2d = workspace_points(1:2, :);
% 计算二维凸包，k_2d是构成凸包边界的点的索引
k_2d = convhull(workspace_points_2d(1,:)', workspace_points_2d(2,:)');
% 获取凸包顶点的二维坐标
hull_vertices_2d = workspace_points_2d(:, k_2d);


% --- 步骤 4: 计算2D凸包的几何中心作为圆心 ---
fprintf('  > 步骤3: 计算轮廓的几何中心作为圆心...\n');
% 使用 polyshape 对象可以方便地计算质心
pgon = polyshape(hull_vertices_2d(1,:)', hull_vertices_2d(2,:)');
[xc, yc] = centroid(pgon);
fprintf('  > 规划的圆心 (X,Y) 坐标为: (%.3f, %.3f)\n', xc, yc);


% --- 步骤 5: 计算圆心到边界的最短距离作为半径 ---
fprintf('  > 步骤4: 计算最大安全半径...\n');
vertices = pgon.Vertices;
num_vertices = size(vertices, 1);
min_dist_sq = inf; % 使用距离的平方进行比较，避免开方运算

% 遍历多边形的每一条边
for i = 1:num_vertices
    p1 = vertices(i, :);
    p2 = vertices(mod(i, num_vertices) + 1, :); % 下一个顶点，实现环绕
    
    % 计算从圆心(xc,yc)到线段p1-p2的最短距离
    line_vec = p2 - p1;
    point_vec = [xc, yc] - p1;
    line_len_sq = sum(line_vec.^2);
    
    if line_len_sq == 0 % 如果p1和p2是同一个点
        dist_sq = sum(point_vec.^2);
    else
        % 将圆心向量投影到边向量上
        t = dot(point_vec, line_vec) / line_len_sq;
        t_clamped = max(0, min(1, t)); % 将投影点限制在线段内部
        
        % 计算线段上离圆心最近的点
        closest_point = p1 + t_clamped * line_vec;
        dist_sq = sum(([xc, yc] - closest_point).^2);
    end
    
    if dist_sq < min_dist_sq
        min_dist_sq = dist_sq;
    end
end
radius = sqrt(min_dist_sq);
fprintf('  > 规划的最大半径为: %.3f\n', radius);


% --- 步骤 6: 组合最终参数 ---
circle_center = [xc; yc; z_optimal];
start_point = [xc + radius; yc; z_optimal];

fprintf('圆形轨迹规划完成！\n');

end