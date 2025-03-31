mainFolder = fileparts(mfilename('fullpath'));
% 添加主文件夹及其所有子文件夹到路径
addpath(genpath(mainFolder));
%% 参数设置
time_step = 3;
state_window = 25:36;
predict_step = 200;
loss_pred_step = 5;
target_dimensions = 68;
epoch = 1000;
lift_function = @lko_gcn_expansion;
test_path = 'data\BellowData\rawData\testData';
model_path = ['models\LKO_GCN_delay3pred5H32P68_network\gcn_network_epoch' num2str(epoch) '.mat'];
koopman_operator_path = ['models\LKO_GCN_delay3pred5H32P68_network\gcn_KoopmanMatrix_epoch' num2str(epoch) '.mat'];
norm_params_path = 'models\LKO_GCN_delay3pred5H32P68_network\norm_params';
control_var_name = 'U_list'; 
state_var_name = 'X_list';    
save_path = ['results\lko_gcn\loss_pred_step' num2str(loss_pred_step) 'dimension' num2str(target_dimensions)] ;

%% 加载训练数据
% 获取所有.mat文件列表
file_list = dir(fullfile(test_path, '*.mat'));
num_files = length(file_list);

% 初始化三维存储数组
control_sequences = [];  % c x N
state_sequences = [];    % dm x N

% 处理数据
for file_idx = 1:num_files
    % 加载数据
    file_path = fullfile(test_path, file_list(file_idx).name);
    data = load(file_path);
    % 合并数据
    control_sequences = cat(2, control_sequences, data.(control_var_name));
    state_sequences = cat(2, state_sequences, data.(state_var_name));
end

% 归一化数据
load_params = load(norm_params_path);
[norm_control, ~] = normalize_data(control_sequences, load_params.params_control);
[norm_state, ~] = normalize_data(state_sequences, load_params.params_state);


% 生成时间延迟数据
[control_timedelay, state_timedelay, label_timedelay] = ...
    generate_gcn_data(norm_control, norm_state, time_step, loss_pred_step); 
%% 加载lko-gcn模型
loadednet =load(model_path);
net = loadednet.net;
gcn_koopman_operator = load(koopman_operator_path);
B = gcn_koopman_operator.B;
A = gcn_koopman_operator.A;

%% 预测
Y_pred = zeros(12, predict_step);
current_state = dlarray(state_timedelay(:, :, :, 1), "SSCB");
for i=1:predict_step
    current_control = dlarray(reshape(control_timedelay(:,1,i),[],1), "CB");
    current_phi_pred = forward(net, current_state, current_control);
    current_state = dlarray(reshape(current_phi_pred(1:36,:),6,6,1,1),'SSCB');
    Y_pred(:,i) = current_phi_pred(25:36,:);
end


Y_true = [squeeze(label_timedelay(:,5,1,1:predict_step)); squeeze(label_timedelay(:,6,1,1:predict_step))];
RMSE = calculateRMSE(Y_pred, Y_true);
disp(['LKO-GCN' 'epoch' num2str(epoch) '的均方根误差是:', num2str(RMSE)])



%% 绘图
% Y_true 是 12×t 的真实值矩阵
% Y_pred 是 12×t 的预测值矩阵

figure('Units', 'normalized', 'Position', [0.1 0.1 0.8 0.8]); % 全屏大窗口
time = 1:size(Y_true, 2); % 生成时间轴

% 绘制12个子图（3行×4列）
for i = 1:12
    subplot(3, 4, i);

    % 绘制真实值和预测值曲线
    plot(time, Y_true(i,:), 'b-', 'LineWidth', 1.5); hold on;
    plot(time, Y_pred(i,:), 'r--', 'LineWidth', 1.5);

    % 美化图形
    title(['Dimension ', num2str(i)]);
    xlabel('Time'); 
    ylabel('Value');
    grid on;

    % 只在第一个子图显示图例
    if i == 1
        legend('True', 'Predicted', 'Location', 'northoutside');
    end

    % 统一坐标轴范围（可选）
    % ylim([min(Y_true(:)), max(Y_true(:))]);
end

% 调整子图间距
set(gcf, 'Color', 'w'); % 设置背景为白色
ha = findobj(gcf, 'type', 'axes');
set(ha, 'FontSize', 9); % 统一字体大小
sgtitle('True vs Predicted Values across 12 Dimensions'); % 总标题

% 保存图像（可选）
saveas(gcf, save_path);