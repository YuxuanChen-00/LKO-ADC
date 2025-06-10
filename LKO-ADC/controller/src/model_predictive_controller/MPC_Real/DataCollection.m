%% 载入数据采集卡
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

%% 参数设置
load_path = '..\..\Data\InputData2\testData';
save_path = '..\..\Data\MotionData2\SorotokiMotionData_test.mat';
num_tracker = 3;
num_state = (num_tracker-1)*6;
lastAoData = [0,0,0,0,0,0,0];
weight = [1,2,3,4,5]';

% 采样频率是控制频率的五倍，每四个点做一次平均作为当前状态
control_freq = 10;  % 控制频率10Hz
sampling_freq = 100;  % 采样频率100Hz
controlRate = robotics.Rate(control_freq);  % 控制更新速率
samplingRate = robotics.Rate(sampling_freq);  % 采样更新速率
num_samples = 5;  % 每3个采样点做平均

%% 加载文件夹下的所有控制输入
file_list= dir(fullfile(load_path, '*.mat'));
num_files = length(file_list);
signal = [];
for file_idx = 1:num_files
    file_path = fullfile(load_path, file_list(file_idx).name);
    control_signal = load(file_path);
    signal = cat(2, signal, control_signal.final_signal);
end
% 初始化数组记录当前时刻位置和控制输入
raw_data = [];
signal_length = size(signal, 2);  % 数据长度
state = zeros(num_state, signal_length);
input = zeros(6, signal_length); 

% 获得初始旋转矩阵和初始点位置
[initRotationMatrix, initPosition] = getInitState(onemotion_data);
last_sample = transferVicon2Base(onemotion_data, initRotationMatrix, initPosition);


%% 控制循环
tic;
for k = 1:signal_length
   [current_state, current_raw, last_sample] = sampleAndFilterViconData(samplingRate, ...
       num_samples, initRotationMatrix, initPosition, weight, last_sample);
    
    % 控制操作
    AoData = [signal(:,k)', 0];
    max_input = [5,5,5,5,5,5,0];
    AoData = min(AoData, max_input);
    linearPressureControl(AoData, lastAoData, samplingRate, instantAoCtrl,...
        scaleData,AOchannelStart, AOchannelCount)
    lastAoData = AoData;
    state(:, k) = current_state;
    input(:,k) = AoData(1:6)';
    raw_data = [raw_data, current_raw];

    % 控制频率更新
    waitfor(controlRate);
    
    time_elapsed = toc;
    disp(['第' num2str(k) '次循环, 用时' num2str(time_elapsed) '秒']);
    tic;
end

save(save_path, 'raw_data', 'state', 'input');



