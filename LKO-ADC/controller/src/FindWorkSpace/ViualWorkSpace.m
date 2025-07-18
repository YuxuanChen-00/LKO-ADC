% =========================================================================
% === 机械臂工作空间可视化脚本 (变量名已更新为 'state') ===
% =========================================================================

clear; clc; close all;

% --- 1. 设置文件夹路径 ---
% !!!【请修改这里】!!!
% 指定包含.mat轨迹文件的文件夹路径。
% 例如: 'D:\Data\RobotTrajectories'
folderPath = '..\..\data\MotionData8\FilteredDataPos\80minTrain'; 

% --- 2. 获取所有.mat文件 ---
% 构建文件搜索模式，找到文件夹下所有.mat文件
filePattern = fullfile(folderPath, '*.mat');
matFiles = dir(filePattern);

% 检查是否找到了任何文件
if isempty(matFiles)
    error('错误：在指定的文件夹 "%s" 中没有找到任何 .mat 文件。', folderPath);
end

% --- 3. 初始化数据存储矩阵 ---
% 初始化两个空矩阵，用于汇集所有文件中的两个关节的坐标数据
all_points_joint1 = [];
all_points_joint2 = [];

% --- 4. 循环加载文件并提取数据 ---
fprintf('开始加载轨迹文件...\n');
for i = 1:length(matFiles)
    baseFileName = matFiles(i).name;
    fullFileName = fullfile(folderPath, baseFileName);
    fprintf('  > 正在处理文件: %s\n', baseFileName);
    
    % 将.mat文件加载到结构体 `loaded_data` 中
    loaded_data = load(fullFileName);
    
    % 根据您的信息，我们知道.mat文件中的轨迹变量名为 'state'
    expectedVarName = 'state';
    
    if isfield(loaded_data, expectedVarName)
        % 从结构体中根据变量名 'state' 提取数据
        trajectory = loaded_data.(expectedVarName);
        
        % 检查轨迹维度是否正确 (6xN)
        if size(trajectory, 1) == 6
            % 提取关节1的坐标 (前3行)
            points_joint1 = trajectory(1:3, :);
            % 提取关节2的坐标 (后3行)
            points_joint2 = trajectory(4:6, :);

            % 将当前文件的数据追加到汇总矩阵中
            all_points_joint1 = [all_points_joint1, points_joint1];
            all_points_joint2 = [all_points_joint2, points_joint2];
        else
            fprintf('    ! 警告: 文件 %s 中的轨迹数据维度不是6，已跳过。\n', baseFileName);
        end
    else
        % 如果找不到名为'state'的变量，则给出警告
        fprintf('    ! 警告: 在文件 %s 中未找到名为 "%s" 的变量，已跳过。\n', baseFileName, expectedVarName);
        fprintf('      该文件包含的变量有: %s\n', strjoin(fieldnames(loaded_data), ', '));
    end
end
fprintf('所有文件加载完毕。\n');

% --- 5. 可视化工作空间 ---
if isempty(all_points_joint1) && isempty(all_points_joint2)
    fprintf('错误: 未能从任何文件中加载有效数据，无法绘图。\n');
    return;
end

fprintf('正在绘制工作空间...\n');
figure('Name', '机械臂工作空间可视化', 'NumberTitle', 'off'); % 创建一个新的图形窗口
hold on; % 允许在同一张图上绘制多个数据集

% 绘制关节1的工作空间散点图
if ~isempty(all_points_joint1)
    scatter3(all_points_joint1(1,:), all_points_joint1(2,:), all_points_joint1(3,:), ...
             15, 'b', 'filled', 'DisplayName', '关节1 工作空间'); % 使用蓝色实心点
end

% 绘制关节2的工作空间散点图
if ~isempty(all_points_joint2)
    scatter3(all_points_joint2(1,:), all_points_joint2(2,:), all_points_joint2(3,:), ...
             15, 'r', 'filled', 'DisplayName', '关节2 工作空间'); % 使用红色实心点
end

% --- 6. 设置图形属性 ---
title('机械臂工作空间可视化', 'FontSize', 16);
xlabel('X 轴', 'FontSize', 12);
ylabel('Y 轴', 'FontSize', 12);
zlabel('Z 轴', 'FontSize', 12);
legend('show'); % 显示图例
grid on;        % 显示网格
axis equal;     % 设置坐标轴比例相等，以真实反映空间形状
view(3);        % 设置为标准三维视角
rotate3d on;    % 允许用鼠标旋转视图
hold off;

fprintf('绘图完成！您现在可以交互式地旋转和缩放三维视图。\n');

% --- 7. 寻找合适的轨迹 ---

[circle_center1, circle_radius1, circle_start_point1] = planTrajectory(all_points_joint1);
[circle_center2, circle_radius2, circle_start_point2] = planTrajectory(all_points_joint2);
% disp(['关节1的最佳轨迹为: ', num2str(circle_center1), num2str(circle_radius1)]);
% disp(['关节2的最佳轨迹为: ', num2str(circle_center2), num2str(circle_radius2)]);