function trajectory = generateReferenceCircle(initial_pos, radius, steps)
% 生成以z轴为法向量的圆轨迹，姿态保持与z轴夹角alpha
% 输入：
%   initial_pos: [x0, y0, z0] 初始位置（基坐标系）
%   radius: 参考圆的半径
%   alpha: 与z轴的固定夹角（弧度）
% 输出：
%   trajectory: Nx6矩阵，每行包括位置[x, y, z]和姿态[nx, ny, nz]，其中
%               nx, ny, nz是末端z轴与基座標系x、y、z轴夹角的余弦值

theta = linspace(0, 2*pi, steps); % 生成100个点的角度参数

% 计算圆心坐标（假设初始位置在theta=0处）
xc = initial_pos(1);
yc = initial_pos(2);
zc = initial_pos(3);

% 生成位置轨迹
x = xc + radius * cos(theta);
y = yc + radius * sin(theta);
z = zc * ones(size(theta));

% 合并轨迹数据
trajectory = [x; y; z];

end