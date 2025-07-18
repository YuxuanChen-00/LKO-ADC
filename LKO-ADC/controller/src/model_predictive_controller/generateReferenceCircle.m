function trajectory = generateReferenceCircle(center, radius_or_start, steps)
% generateReferenceCircle: 在三维空间中生成一个圆形参考轨迹。
%
% 输入参数:
%   center: 一个包含3个元素的向量 [xc, yc, zc]，指定圆心的坐标。
%   radius_or_start:
%       - 一个标量（单个数字），代表圆的半径。
%       - 一个包含3个元素的向量 [xs, ys, zs]，代表圆轨迹的起始点。
%   steps: 为轨迹生成的点的数量。
%
% 输出:
%   trajectory: 一个 3xN 的矩阵 (N = steps)，代表圆上点的 [x; y; z] 坐标。

% 提取圆心坐标
xc = center(1);
yc = center(2);
zc = center(3);

% 检查第二个参数是半径还是起始点
if isscalar(radius_or_start)
    % 情况1：第二个参数是标量（半径）
    radius = radius_or_start;
    
    % 生成角度。圆将从标准的0弧度位置（3点钟方向）开始。
    theta = linspace(0, 2*pi, steps);
    
else
    % 情况2：第二个参数是向量（起始点）
    start_point = radius_or_start;
    xs = start_point(1);
    ys = start_point(2);
    % 注意：我们假设起始点的z坐标与圆心定义的平面一致。
    % 我们使用xs和ys在XY平面上定义圆。
    
    % 将圆心到起始点的欧几里得距离计算为半径
    radius = sqrt((xs - xc)^2 + (ys - yc)^2);
    
    % 处理半径为零的边缘情况（起始点即圆心）
    if radius == 0
        x = xc * ones(1, steps);
        y = yc * ones(1, steps);
        z = zc * ones(1, steps);
        trajectory = [x; y; z];
        return;
    end
    
    % 计算给定起始点的初始角度
    theta_start = atan2(ys - yc, xs - xc);
    
    % 从计算出的初始角度开始，生成一个完整的圆
    theta = linspace(theta_start, theta_start + 2*pi, steps);
end

% 为轨迹生成 (x, y, z) 坐标
x = xc + radius * cos(theta);
y = yc + radius * sin(theta);
z = zc * ones(size(theta)); % z坐标是恒定的（圆所在的平面）

% 将坐标合并到输出矩阵中
trajectory = [x; y; z];

end