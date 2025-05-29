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
colorbar
xlabel('时间延迟步长')
ylabel('升维维度')
title('预测误差分布热力图')

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