load('poly_param_report.mat'); % 加载数据表
data = report_table; % 替换为实际变量名

% 创建网格矩阵
delay_steps = 1:10;
dimensions = 6:30;
[xx,yy] = meshgrid(delay_steps, dimensions);
zz = griddata(data.delay, data.dimension, data.mean_rmse, xx, yy);

% 绘制热力图
figure
contourf(xx, yy, zz, 50, 'LineColor','none')
colormap(jet(256))

% 设置坐标轴标签字号
xlabel('时间延迟步长', 'FontSize', 14, 'FontWeight', 'bold')  % 添加字体大小和加粗
ylabel('升维维度', 'FontSize', 14, 'FontWeight', 'bold')

% 设置标题字号
title('预测误差分布热力图', 'FontSize', 16, 'FontWeight', 'bold')

% 设置颜色栏字号
c = colorbar;
c.Label.String = 'RMSE';  % 添加颜色栏标题
c.Label.FontSize = 14;
c.FontSize = 12;  % 设置刻度字号

% 设置坐标轴刻度字号
ax = gca;
ax.FontSize = 12;  % 统一坐标轴刻度数字字号

% 优化图形显示
set(gcf, 'Color', 'white')  % 设置背景为白色
set(gca, 'LineWidth', 1.2)  % 加粗坐标轴线

% 接续前代码
figure
surf(xx, yy, zz, 'EdgeColor','none')
colormap(turbo)
xlabel('时间延迟步长')
ylabel('升维维度')
zlabel('RMSE')
title('三维误差曲面')
view(40,35) % 调整视角

figure
hold on
for delay = 1:10
    idx = data.delay == delay;
    plot(data.dimension(idx), data.mean_rmse(idx),...
        'LineWidth',1.5,...
        'DisplayName',['Delay=' num2str(delay)])
end
xlabel('升维维度')
ylabel('RMSE')
title('不同时间延迟下的误差变化')
legend('Location','northeastoutside')
grid on

figure
scatter3(data.delay, data.dimension, data.mean_rmse,...
    50, data.mean_rmse, 'filled')
colormap(parula)
xlabel('时间延迟')
ylabel('升维维度')
zlabel('RMSE')
title('三维气泡图')
view(25,30) % 最佳观察角度
colorbar