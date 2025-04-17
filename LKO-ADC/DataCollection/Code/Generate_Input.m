%% 参数设置
min_pressure = [0,0,0,0,0,0,0]';
max_pressure = [5,5,5,5,5,5,5]';
t = 30;
fs = 20;
N = t*fs;
D = 7;
T = 0.04;
L = 100; % 重叠长度（根据信号特性调整）
num_samples = 10; % 每种信号生成10个样本
path = '..\Data\InputData\SonimInput_dataset_0.04_3.mat';

%% 信号参数
% 随机游走参数
maxStep = 2;
probRise = 0.2;
probFall = 0.2;
probHold = 0.6;

% 扫频参数
f0_chirp = 0.002;
f1_chirp = 0.005;

% 多频正弦参数
frequencies = [0,0.002;0.002,0.005;0.005,0.01];

% PRBS参数
f0_prbs = 0.1;
f1_prbs = 0.8;
segnum = 5;

%% 生成信号样本（每种类型10个）
signal_types = {'PRBS', 'Chirp', 'MultiSine', 'LHS', 'RandomWalk'};
samples = cell(length(signal_types), num_samples);

for i = 1:num_samples
    samples{1,i} = Generate_SegPRBS(D,N,f0_prbs,f1_prbs,segnum,min_pressure,max_pressure);
    samples{2,i} = Generate_Chirp(D,N,f0_chirp,f1_chirp,min_pressure,max_pressure);
    samples{3,i} = Generate_MultiSine(D,N,frequencies,min_pressure,max_pressure);
    samples{4,i} = Generate_LHS(D,N,T,min_pressure,max_pressure);
    samples{5,i} = Generate_RandomWalk(D,N,T,maxStep,probRise,probFall,probHold,min_pressure,max_pressure);
end

%% 生成组合段并拼接
final_signal = [];
for seg_num = 1:num_samples
    % 按顺序获取五种信号
    segment = [];
    for sig_type = 1:length(signal_types)
        current_sig = samples{sig_type, seg_num};
        
        % 修改调用方式：添加min_pressure和max_pressure参数
        if isempty(segment)
            segment = current_sig;
        else
            segment = smooth_connect(segment, current_sig, L, min_pressure, max_pressure);
        end
    end
    % 组合段间连接也需要传递参数
    if isempty(final_signal)
        final_signal = segment;
    else
        final_signal = smooth_connect(final_signal, segment, L, min_pressure, max_pressure);
    end
end

% %% 保存结果
% save(path, 'final_signal');

%% 修改后的平滑连接函数
function connected = smooth_connect(sig1, sig2, L, min_pressure, max_pressure) % 添加参数
    fade_out = linspace(1, 0, L);
    fade_in = linspace(0, 1, L);
    
    % 获取连接部分
    end_part = sig1(:, end-L+1:end) .* fade_out;
    start_part = sig2(:, 1:L) .* fade_in;
    
    % 拼接信号
    connected = [sig1(:, 1:end-L), (end_part + start_part), sig2(:, L+1:end)];
    
    % 确保幅值限制（现在能识别参数）
    connected = max(min(connected, max_pressure), min_pressure); % 列向量与矩阵的广播运算
end

%% 绘制结果（示例通道）
figure;
for i = 1:7
    subplot(7,1,i);
    plot(final_signal(i,:));
    title(['Channel ', num2str(i)]);
    ylim([min_pressure(i), max_pressure(i)]);
end