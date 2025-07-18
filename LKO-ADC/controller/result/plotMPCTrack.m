
% 获取当前文件所在目录
currentDir = fileparts(mfilename('fullpath'));

% 获取上一级目录
parentDir = fileparts(currentDir);

% 只添加上一级目录本身（不包括其子目录）
addpath(parentDir);

%% 加载数据
% data = load('MPC\60secCircleTrack_payload_25g.mat');
% data = load('MPC\60secSquareTrack_payload_25g.mat');
% data = load('MPC\60secLemniscateTrack_payload_25g.mat');
% data = load('MPC\60secCircleTrack_payload_10g.mat');
% data = load('MPC\60secSquareTrack_payload_10g.mat');
% data = load('MPC\60secLemniscateTrack_payload_10g.mat');
% data = load('MPC\60secCircleTrack.mat');
% data = load('MPC\60secSquareTrack.mat');
data = load('MPC\60secLemniscateTrack.mat');


% data = load('Adaptive\60secCircleTrack_payload_25g.mat');
% data = load('Adaptive\60secSquareTrack_payload_25g.mat');
% data = load('Adaptive\60secLemniscateTrack_payload_25g.mat');
% data = load('Adaptive\60secCircleTrack_payload_10g.mat');
% data = load('Adaptive\60secSquareTrack_payload_10g.mat');
% data = load('Adaptive\60secLemniscateTrack_payload_10g.mat');
% data = load('Adaptive\60secCircleTrack.mat');
% data = load('Adaptive\60secSquareTrack.mat');
% data = load('Adaptive\60secLemniscateTrack.mat');


Y_ref = data.Y_ref;
Y_history = data.Y_history;
U_history = data.U_history;
k_steps = size(Y_history, 2); 

%% 计算均方根误差
mse1 = calculateRMSE(Y_ref(1:3, 101:k_steps), Y_history(1:3,101:end));
mse2 = calculateRMSE(Y_ref(4:6, 101:k_steps), Y_history(4:6,101:end));
fprintf('第一关节轨迹跟踪均方根误差为: %2f, 第二关节轨迹跟踪的均方根误差为: %2f\n', mse1, mse2);


% %% 绘制结果
% time_vec = 1:k_steps; % 时间向量，对应 Y_history 和 X_koopman_history
% time_vec_input = 1:k_steps-1;
% 
% % 1. 3D轨迹跟踪效果 (假设Y的前三维是x,y,z坐标)
% figure;
% plot3(Y_history(1,:), Y_history(2,:), Y_history(3,:), 'b-', 'LineWidth', 1.5);
% hold on;
% plot3(Y_ref(1,1:k_steps), Y_ref(2,1:k_steps), Y_ref(3,1:k_steps), 'r--', 'LineWidth', 1.5);
% plot3(Y_history(1,1), Y_history(2,1), Y_history(3,1), 'go', 'MarkerSize', 10, 'MarkerFaceColor', 'g'); % 起点
% plot3(Y_ref(1,1), Y_ref(2,1), Y_ref(3,1), 'ro', 'MarkerSize', 10, 'MarkerFaceColor', 'r'); % 参考起点
% xlabel('X position');
% ylabel('Y position');
% zlabel('Z position');
% title('3D Trajectory Tracking');
% % legend('Actual Trajectory (MPC)', 'Reference Trajectory', 'Actual Start', 'Reference Start');
% axis equal;
% grid on;
% 
% % 2. 3D轨迹跟踪效果 (假设Y的前三维是x,y,z坐标)
% plot3(Y_history(4,:), Y_history(5,:), Y_history(6,:), 'b-', 'LineWidth', 1.5);
% hold on;
% plot3(Y_ref(4,1:k_steps), Y_ref(5,1:k_steps), Y_ref(6,1:k_steps), 'r--', 'LineWidth', 1.5);
% plot3(Y_history(4,1), Y_history(5,1), Y_history(6,1), 'go', 'MarkerSize', 10, 'MarkerFaceColor', 'g'); % 起点
% plot3(Y_ref(4,1), Y_ref(5,1), Y_ref(6,1), 'ro', 'MarkerSize', 10, 'MarkerFaceColor', 'r'); % 参考起点
% xlabel('X position');
% ylabel('Y position');
% zlabel('Z position');
% title('3D Trajectory Tracking');
% legend('Actual Trajectory1', 'Reference Trajectory1', 'Actual Start1', 'Reference Start1', ...
%     'Actual Trajectory2', 'Reference Trajectory2', 'Actual Start2', 'Reference Start2');
% axis equal;
% grid on;
% view(3); % 3D视角
% 
% 
% % 2. Y中每个通道的跟踪效果
% figure;
% output_labels = {'Y_1 (Pos X)', 'Y_1 (Pos Y)', 'Y_1 (Pos Z)','Y_2 (Pos X)', 'Y_2 (Pos Y)', 'Y_2 (Pos Z)'};
% for i = 1:6
%     subplot(6, 1, i); % 假设 n_Output 是偶数，例如6 -> 3x2 subplot
%     plot(time_vec, Y_history(i,:), 'b-', 'LineWidth', 1);
%     hold on;
%     plot(time_vec, Y_ref(i,1:k_steps), 'r--', 'LineWidth', 1);
%     xlabel('Time step (k)');
%     ylabel(output_labels{i});
%     title(['Tracking of Output Channel: ', output_labels{i}]);
%     grid on;
% end
% legend('Actual', 'Reference', 'Location', 'best');
% sgtitle('Output Channel Tracking Performance'); % Super title for all subplots
% 
% % 3. 控制输入U
% figure;
% input_labels = cell(1, 6);
% for i=1:6
%     input_labels{i} = ['U_{', num2str(i), '}'];
% end
% for i = 1:6
%     subplot(ceil(6/2), 2, i); % 调整subplot布局
%     plot(time_vec_input, U_history(i,:), 'm-', 'LineWidth', 1);
%     xlabel('Time step (k)');
%     ylabel(input_labels{i});
%     title(['Control Input: ', input_labels{i}]);
%     grid on;
% end
% sgtitle('Control Inputs U');
% 
% 
% disp('Plotting complete.');