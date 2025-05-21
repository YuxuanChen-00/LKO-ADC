%% 参数设置
min_pressure = [0,0,0,0,0,0]';
max_pressure = [5,5,5,5,5,5]';
t = 20;
fs = 10;
N = t*fs;
D = 6;
T = 0.01;
L = 10; % 重叠长度（根据信号特性调整）
num_samples = 1; % 每种信号生成10个样本
path = ['..\..\Data\InputData2\testData\SorotokiInputData_test_' num2str(T) '.mat'];

%% 信号参数
% 随机游走参数
maxStep = 2;
probRise = 0.2;
probFall = 0.2;
probHold = 0.6;

% 扫频参数
f0_chirp = T*0.04;
f1_chirp = T*0.1;

% 多频正弦参数
frequencies = T*[0,0.05;0.05,0.1;0.1,0.2];

% PRBS参数
f0_prbs = T*1;
f1_prbs =T*3.5;
segnum = 2;

% 阶跃信号参数
holdSteps = 20;
interpSteps = 30;
%% 生成信号样本（每种类型10个）
signal_types = {'PRBS', 'Chirp', 'MultiSine', 'LHS', 'RandomWalk'};
samples = cell(length(signal_types), num_samples);

for i = 1:num_samples
    samples{1,i} = generateSegPRBS(D,N,f0_prbs,f1_prbs,segnum,min_pressure,max_pressure);
    samples{2,i} = generateChirp(D,N,f0_chirp,f1_chirp,min_pressure,max_pressure);
    samples{3,i} = generateMultiSine(D,N,frequencies,min_pressure,max_pressure);
    samples{4,i} = generateLHS(D,N,T,min_pressure,max_pressure);
    samples{5,i} = generateRandomWalk(D,N,T,maxStep,probRise,probFall,probHold,min_pressure,max_pressure);
%     samples{6,i} = generateStep(D, max_pressure, min_pressure, interpSteps, holdSteps);
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

final_signal = [final_signal, zeros(6,50)];

% %% 保存结果
save(path, 'final_signal');


%% 绘制结果（示例通道）
figure;
for i = 1:D
    subplot(D,1,i);
    plot(final_signal(i,:));
    title(['Channel ', num2str(i)]);
    ylim([min_pressure(i), max_pressure(i)]);
end


%% 计算各维度最大步长及位置（添加在绘图代码前）
max_steps = zeros(D,1);    % 存储最大步长
max_positions = zeros(D,1); % 存储对应位置索引

for i = 1:D
    % 计算相邻点差值的绝对值
    diffs = abs(diff(final_signal(i,:)));
    
    % 查找最大值及其位置
    [max_val, max_idx] = max(diffs);
    
    % 记录结果（位置对应原始信号中的起始点索引）
    max_steps(i) = max_val;
    max_positions(i) = max_idx; % 这个位置是diff后的索引，对应原信号中max_idx到max_idx+1的跳跃
end

%% 显示详细结果
fprintf('\n== 各维度最大步长及位置 ==\n');
disp(array2table([(1:D)', max_steps, max_positions, max_positions+1],...
    'VariableNames', {'维度','最大步长','起始点','结束点'}));


%% 修改后的平滑连接函数
function connected = smooth_connect(sig1, sig2, L, min_pressure, max_pressure)
    % 使用余弦曲线过渡（更平滑的一阶导数）
    t = linspace(0, 1, L);
    fade_out = (1 + cos(pi*t))/2;  % 从1平滑衰减到0
    fade_in  = (1 - cos(pi*t))/2;  % 从0平滑增长到1
    
    % 其余代码与原函数相同
    end_part  = sig1(:, end-L+1:end) .* fade_out;

    start_part = sig2(:, 1:L) .* fade_in;

    connected = [sig1(:, 1:end-L), (end_part + start_part), sig2(:, L+1:end)];
    connected = max(min(connected, max_pressure), min_pressure);
end