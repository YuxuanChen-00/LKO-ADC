%% 频域分析脚本
% 作者：人工智能助手
% 日期：2023-10-15
% 描述：本脚本用于分析12通道数据各维度的频率成分

%% 参数设置
Fs = 10;              % 采样频率（根据实际情况修改）
low_freq = 0.003;          % 低频截止频率（单位：Hz）
high_freq = 0.01;        % 高频截止频率（单位：Hz）
plot_dimension = 1;     % 选择要绘制频谱图的维度（1-12）

%% 生成示例数据（实际使用时替换为您的数据）
train_path = 'data\SorotokiData\MotionData3_without_Direction\trainData'; % 训练数据路径
test_path = 'data\SorotokiData\MotionData3_without_Direction\testData';   % 测试数据路径

%% 加载训练数据
% 获取所有.mat文件列表
file_list = dir(fullfile(train_path, '*.mat'));
num_files = length(file_list);

% 初始化三维存储数组
control_sequences = [];  % c x N
state_sequences = [];    % dm x N

% 处理数据
for file_idx = 1:num_files
    % 加载数据
    file_path = fullfile(train_path, file_list(file_idx).name);
    data = load(file_path);
    % 合并数据
    control_sequences = cat(2, control_sequences, data.(control_var_name));
    state_sequences = cat(2, state_sequences, data.(state_var_name));
end

t = size(state_sequences,2);
data = state_sequences;
%% 预处理
% 去直流分量
data_centered = data - mean(data, 2);

% 加汉宁窗减少频谱泄漏
window = hanning(t)';   % 生成窗函数
windowed_data = data_centered .* window;

%% 傅里叶变换
nfft = 2^nextpow2(t);   % 优化FFT计算效率
fft_result = fft(windowed_data, nfft, 2);

%% 计算功率谱密度
P2 = abs(fft_result/nfft);      % 双边谱
P1 = P2(:, 1:nfft/2+1);        % 单边谱
P1(:, 2:end-1) = 2*P1(:, 2:end-1); % 调整幅度

%% 创建频率轴
f = Fs*(0:(nfft/2))/nfft;

%% 频率成分分析
% 获取频率索引
low_idx = find(f <= low_freq);
mid_idx = find(f > low_freq & f <= high_freq);
high_idx = find(f > high_freq & f <= Fs/2);

% 计算各频段能量
low_power = sum(P1(:, low_idx).^2, 2);
mid_power = sum(P1(:, mid_idx).^2, 2);
high_power = sum(P1(:, high_idx).^2, 2);
total_power = low_power + mid_power + high_power;

% 计算能量占比
power_distribution = [low_power, mid_power, high_power] ./ total_power * 100;

%% 可视化结果
figure('Color','white','Position',[100 100 1200 800])

% 绘制能量分布柱状图
subplot(2,2,[1,2])
bar(power_distribution, 'stacked')
title('各维度频率成分能量分布')
xlabel('数据维度')
ylabel('能量占比 (%)')
legend(['低频 (0-',num2str(low_freq),'Hz)'],...
       ['中频 (',num2str(low_freq),'-',num2str(high_freq),'Hz)'],...
       ['高频 (',num2str(high_freq),'-',num2str(Fs/2),'Hz)'])
xticks(1:12)
grid on

% 绘制选定维度的频谱图
subplot(2,2,3)
plot(f, P1(plot_dimension,:))
title(['维度 ',num2str(plot_dimension),' 的频谱'])
xlabel('频率 (Hz)')
ylabel('幅度')
xlim([0 Fs/2])
grid on

% 绘制所有维度平均频谱
subplot(2,2,4)
semilogy(f, mean(P1,1))
title('所有维度平均频谱')
xlabel('频率 (Hz)')
ylabel('对数幅度')
xlim([0 Fs/2])
grid on

%% 显示数值结果
disp('=== 各维度频率能量占比（%）===')
disp('   低频    中频    高频')
disp(round(power_distribution,1))