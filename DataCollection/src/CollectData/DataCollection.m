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
load_path = '..\Data\InputData\SorotokiInputData.mat';
save_path = '..\Data\MotionData\SorotokiMotionData.mat';
input_data = load(load_path);
signal = input_data.signal_random_input;
num_input = size(signal,1);  % 输入通道数
num_tracker = 3;
num_state = num_tracker*2;
length = size(signal, 2);  % 数据长度
lastAoData = [0,0,0,0,0,0,0];
weight = [1,2,3,4,5]';

% 采样频率是控制频率的五倍，每四个点做一次平均作为当前状态
control_freq = 10;  % 控制频率20Hz
sampling_freq = 100;  % 采样频率80Hz
controlRate = robotics.Rate(control_freq);  % 控制更新速率
samplingRate = robotics.Rate(sampling_freq);  % 采样更新速率

num_samples = 5;  % 每5个采样点做平均
sample_buffer = zeros(num_state, num_samples);  % 用于存储采样值的缓冲区
sample_index = 1;  % 缓冲区索引

% 获得初始旋转矩阵和初始点位置
initRotationMatrix = getInitRotationMatrix(onemotion_data);
last_sample = transferVicon2Base(onemotion_data, initRotationMatrix);

% 初始化数组记录当前时刻位置和控制输入
raw_data = [];
state = zeros(num_state, length);
input = zeros(num_input, length); 

for k = 1:length
   [current_state, current_raw, last_sample] = sampleAndFilterViconData(samplingRate, ...
       num_samples, num_state, initRotationMatrix, weight, last_sample);
    
    % 控制操作
    AoData = [signal(:,k)', 0];
    max_input = [5,5,5,5,5,5,0];
    AoData = min(AoData, max_input);
    linearPressureControl(AoData, lastAoData, samplingRate, instantAoCtrl_1,...
        scaleData,AOchannelStart, AOchannelCount)
    lastAoData = AoData;
    state(:, k) = current_state;
    input(:,k) = AoData';
    raw_data = [raw_data, current_raw];

    % 控制频率更新
    waitfor(controlRate);
end

save(save_path, 'raw_data', 'state', 'input');



