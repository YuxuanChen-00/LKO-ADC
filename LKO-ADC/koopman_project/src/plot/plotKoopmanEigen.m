% 定义矩阵并计算特征值
% A = [5 -5; -2 5];
[V, D] = eig(A);
eigenvalues = diag(D);

% 绘制单位圆
theta = linspace(0, 2*pi, 100);
x = cos(theta);
y = sin(theta);

figure;
plot(x, y, 'k--', 'LineWidth', 1.5);
hold on;
scatter(real(eigenvalues), imag(eigenvalues), 'ro', 'filled');
axis equal; grid on;
xlabel('实部'); ylabel('虚部');
title('矩阵特征值在单位圆上的分布');
legend('单位圆', '特征值');