% 假设您的数据已经加载到名为 'my_results' 的 struct 数组中
% 如果数据存储在 .mat 文件中，请先加载它：
% my_results = load('result8_lstm_koopman\full_results.mat');
my_results = load('result7_without_td_prev\full_results.mat');
my_results = my_results.sorted_results;


% --- 1. 提取并识别所有唯一的 delay 值 ---
% [my_results.delay] 会将所有元素的 delay 字段提取成一个向量
all_delays = [my_results.delay];
unique_delays = unique(all_delays);
num_delays = length(unique_delays);

% --- 2. 准备绘图窗口 ---
figure;       % 创建一个新的图形窗口
hold on;      % 允许在同一张图上绘制多条曲线

% 使用 colormap 来为不同的曲线自动选择颜色
colors = lines(num_delays);

% --- 3. 循环遍历每个 delay 值并绘图 ---
for i = 1:num_delays
    current_delay = unique_delays(i);
    
    % a. 找到当前 delay 对应的所有数据点的索引
    indices = find(all_delays == current_delay);
    
    % b. 提取对应的 dimension 和 mean_rmse 数据
    current_dimensions = [my_results(indices).dimension];
    current_rmses = [my_results(indices).mean_rmse];
    
    % c. (重要) 按 dimension 对数据进行排序，以确保曲线连接正确
    [sorted_dimensions, sort_order] = sort(current_dimensions);
    sorted_rmses = current_rmses(sort_order);
    
    % d. 绘制当前 delay 对应的曲线
    plot(sorted_dimensions, sorted_rmses, ...
         'Marker', 'o', ...                      % 数据点样式
         'LineWidth', 1.5, ...                   % 线条宽度
         'Color', colors(i, :), ...              % 设置线条颜色
         'DisplayName', ['Delay = ' num2str(current_delay)]); % 设置图例名称
end

% --- 4. 美化和完善图形 ---
hold off;     % 结束在当前图上绘图

% 添加标题和坐标轴标签
title('timedelay-poly koopman prediction', 'FontSize', 14);
xlabel('Dimension', 'FontSize', 12);
ylabel('Mean RMSE', 'FontSize', 12);

% 显示图例
legend('show', 'Location', 'best'); % 'Location', 'best' 会自动选择最佳位置

% 添加网格线
grid on;

% 保存图像到文件
% saveas(gcf, 'rmse_vs_dimension_plot.png');

disp('绘图完成！');