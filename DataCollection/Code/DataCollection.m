delete (instrfindall);
serialforce = serial('COM2', 'BaudRate', 115200, 'Parity', 'none',...
                'DataBits', 8, 'StopBits', 1);
deviceDescription = 'PCI-1716,BID#0'; % Analog input card
deviceDescription_1 = 'PCI-1727U,BID#0'; % Analog input card  
AOchannelStart = int32(1);
AOchannelCount = int32(7);  
BDaq = NET.addAssembly('Automation.BDaq4');
errorCode = Automation.BDaq.ErrorCode.Success;
instantAoCtrl = Automation.BDaq.InstantAoCtrl();
instantAoCtrl.SelectedDevice = Automation.BDaq.DeviceInformation(...
    deviceDescription_1);
scaleData = NET.createArray('System.Double', int32(64));

global onemotion_data;
serialvicon = serial('COM1');
set(serialvicon,'BaudRate',115200);
set(serialvicon,'BytesAvailableFcnMode','Terminator'); 
set(serialvicon,'Terminator','LF');
set(serialvicon,'BytesAvailableFcn',{@ReceiveVicon});
fopen(serialvicon);

% 加载输入数据
load_path = '..\Data\InputData\SonimInput_dataset_0.01_1.mat';
save_path = '..\Data\MotionData\RandomInput\random_input_0.01_1.mat';
input_data = load(load_path);
signal = input_data.signal_random_input;
num_input = size(signal,1);  % 输入通道数
num_state = 18;
length = size(signal, 2);  % 数据长度

% 采样频率是控制频率的五倍，每四个点做一次平均作为当前状态
control_freq = 20;  % 控制频率20Hz
sampling_freq = 80;  % 采样频率80Hz
controlRate = robotics.Rate(control_freq);  % 控制更新速率
samplingRate = robotics.Rate(sampling_freq);  % 采样更新速率

num_samples = 4;  % 每4个采样点做平均
sample_buffer = zeros(num_state, num_samples);  % 用于存储采样值的缓冲区
sample_index = 1;  % 缓冲区索引

% 获得初始旋转矩阵
initRotationMatrix = getInitRotationMatrix(onemotion_data);

% 初始化数组记录当前时刻位置和控制输入
raw_data = zeros(24, num_samples*length);
p = 1;

raw_state = zeros(num_state, length);
state = zeros(num_state, length);
input = zeros(num_input, length); 
median_window_size = 7;
median_window = zeros(num_state, median_window_size);
for k = 1:length
    % 进行4次采样，更新采样缓冲区
    for i = 1:num_samples
        waitfor(samplingRate);  % 按照80Hz的频率进行采样
        
        % 获取一个新的采样点
        new_sample = Raw2Xmeas(onemotion_data, initRotationMatrix);

        raw_data(:,p) = onemotion_data;
        p = p+1;
        % 将新采样值存入缓冲区
        sample_buffer(:, sample_index) = new_sample;
        sample_index = sample_index + 1;  % 更新索引
        
        % 如果缓冲区已满，重新从头开始
        if sample_index > num_samples
            sample_index = 1;
        end
    end
    
    % 采样区间异常点检测
    weight = [0.1, 0.2, 0.3, 0.4]';
    weight = weight/sum(weight);
    
    % 计算当前状态（4个采样点的加权平均值）
    current_state = sample_buffer*weight;  % 计算当前状态（4个采样点的平均值）
    if any(isnan(current_state))
        disp('当前vicon采样缺失，重新采集数据')
        k = k-1;
        continue;
    end
    raw_state(:,k) = current_state;
    
    % 当前状态异常点检测
    threshold = [10,10,10,0.05,0.05,0.05,...
        20,20,20,0.15,0.15,0.15,...
        30,30,30,0.2,0.2,0.2];
    
    if k > median_window_size
        % median_window = [median_window(:,2:median_window_size), current_state];
        median_window = state(:,k-median_window_size:k-1);
        current_state = median_filter(current_state, state(:,k-1), median_window, threshold);
    end
    
    % 控制操作
    AoData = signal(:,k)';
    max_input = [5,5,5,5,5,5,2];
    AoData = min(AoData, max_input);
    AoWrite(AoData,instantAoCtrl,scaleData,AOchannelStart,AOchannelCount);
    
    state(:, k) = current_state;
    input(:,k) = AoData';

    % 控制频率更新
    waitfor(controlRate);
end

save(save_path, 'raw_data', 'state', 'input');
save('raw_data.mat','raw_data');


figure(1)
for p = 1:6
    subplot(6,1,p);
    plot(state(p,:));
end

figure(2)
for p = 7:12
    subplot(6,1,p-6);
    plot(state(p,:));
end

figure(3)
for p = 13:18
    subplot(6,1,p-12);
    plot(state(p,:));
end

% figure(4)
% for p = 1:6
%     subplot(6,1,p);
%     plot(raw_state(p,:));
% end
% 
% figure(5)
% for p = 7:12
%     subplot(6,1,p-6);
%     plot(raw_state(p,:));
% end
% 
% figure(6)
% for p = 13:18
%     subplot(6,1,p-12);
%     plot(raw_state(p,:));
% end
