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


% 设置采样频率
samplerate = 0.3;
Rate = robotics.Rate(samplerate);

samplerate=10;
Rate = robotics.Rate(samplerate);
reset(Rate);

AoData1 = [0, 5, 5, 0, 0, 0, 0];
init_rotation_matrix = getInitRotationMatrix(onemotion_data);

n_step = 100;
raw_data = zeros(24, n_step*2);
X_meas = zeros(18, n_step*2);
input_data = zeros(7,n_step*2);

for i = 1:n_step
    AoData = AoData1*1/n_step*i;
    AoWrite(AoData,instantAoCtrl,scaleData,AOchannelStart,AOchannelCount);
    waitfor(Rate);
    if mod(i,20) == 0
        pause(5);
    end
    raw_data(:,i) = onemotion_data;
    X_meas(:,i) = Raw2Xmeas(onemotion_data, init_rotation_matrix);
    input_data(:,i) = AoData';
end

for i = 1:n_step
    AoData = AoData1 - AoData1*1/n_step*i;
    AoWrite(AoData,instantAoCtrl,scaleData,AOchannelStart,AOchannelCount);
    waitfor(Rate);
    if mod(i,20) == 0
        pause(5);
    end
    raw_data(:, i+n_step) = onemotion_data;
    X_meas(:, i+n_step) = Raw2Xmeas(onemotion_data, init_rotation_matrix);
    input_data(:, i+n_step) = AoData';
end

save_path = '..\Data\StepResponse_bend1.mat';
save(save_path, 'raw_data', 'X_meas', 'input_data');
