
%% --- 1. 数据加载  ---
% 数据格式: channels x samples (通道数 x 样本数)

load_path = 'data\SorotokiData\MotionData3_without_Direction\testData\SorotokiMotionData_test.mat';
save_path = 'data\SorotokiData\Filtered_PositionData\testData\SorotokiMotionData_test.mat';


% --- 加载真实数据部分 ---
loaded_data = load(load_path); % 假设 loaded_data.my_channel_data 是 channels x samples
rawData = loaded_data.state;

[num_channels, num_samples] = size(rawData); % 更新维度
% t = (1:num_samples)'; % 更新时间向量
t = 1:2880;

%% --- 2. 定义滤波器参数 (您可以在这里方便地调整参数) ---
poly_order_g1 = 3;  
frame_len_g1  = 15; 


%% --- 3. 对数据应用滤波器 ---
fprintf('正在应用手动实现的Savitzky-Golay滤波器...\n');
filteredData = zeros(size(rawData)); % 初始化滤波后的数据矩阵 (channels x samples)

for ch = 1:num_channels % 循环遍历每个通道 (即 rawData 的每一行)
    signal_to_filter_row = rawData(ch, :);       % 获取当前通道的原始数据 (行向量)
    signal_to_filter_col = signal_to_filter_row'; % 转置为列向量以供滤波器函数使用
    
    
    fprintf('通道 %d: 应用参数 (阶数: %d, 帧长: %d)\n', ch, poly_order_g1, frame_len_g1);
    filtered_signal_col = apply_sg_filter(signal_to_filter_col, poly_order_g1, frame_len_g1);

    
    filteredData(ch, :) = filtered_signal_col'; % 将滤波后的列向量转置回行向量并存储
end
fprintf('滤波处理完成。\n');


%% --- 4. 绘制滤波前后的效果图 ---
fprintf('正在绘制结果图表...\n');
figure('Name', '手动SG滤波效果对比 (channels x samples)', 'Units', 'normalized', 'OuterPosition', [0 0 1 1]); 

num_plot_rows = 3; 
num_plot_cols = 2;

for ch = 1:num_channels
    subplot(num_plot_rows, num_plot_cols, ch); 
    
    % rawData(ch,:) 是行向量，t 是列向量。绘图时需要将数据转置为列向量。
    plot(t, rawData(ch, t)', 'Color', [0.7 0.7 0.95],'LineWidth', 2, 'DisplayName', '原始信号'); 
    hold on; 
    plot(t, filteredData(ch, t)', 'r', 'LineWidth', 1.2, 'DisplayName', '滤波后 (手动SG)'); 
    hold off; 
    
    title(sprintf('通道 %d', ch));         
    xlabel('时间 / 样本索引');             
    ylabel('幅值');                       
    legend('show', 'Location', 'best'); 
    grid on;                            
    axis tight;                         
end

if exist('sgtitle', 'file') 
    sgtitle('12通道信号 (channels x samples) - 手动Savitzky-Golay滤波前后对比');
else
    disp('提示: 当前MATLAB版本可能不支持 sgtitle 函数。');
end

fprintf('绘图完成。请查看生成的图形窗口。\n');

%% --- 5. 保存滤波后的数据 ---
state = filteredData;
input = loaded_data.input;
raw_data = loaded_data.raw_data;
save(save_path, 'state', 'input', 'raw_data')

