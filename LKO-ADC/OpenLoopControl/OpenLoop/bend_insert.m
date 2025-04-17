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

samplerate=20;
Rate = robotics.Rate(samplerate);
reset(Rate);
step = 60;

Rate2 = robotics.Rate(0.25);
reset(Rate2);

AoData = [ 4, 0, 0, 0, 4, 0, 1]; % 对应气压

% 弯曲第一节
for i =1:step
    current_pressure = AoData(1)/step*i;
    current_pressure = min(current_pressure, 5);
    scaleData.Set(0,current_pressure);
    errorCode = instantAoCtrl.Write(AOchannelStart, AOchannelCount, scaleData);
    waitfor(Rate);
end

% 弯曲第二节
for i =1:step
    current_pressure = AoData(5)/step*i;
    disp(current_pressure);
    current_pressure = min(current_pressure,6);
    scaleData.Set(4,current_pressure);
    errorCode = instantAoCtrl.Write(AOchannelStart, AOchannelCount, scaleData);
    waitfor(Rate);
end

waitfor(Rate2);
waitfor(Rate2);
% 伸长第三节
current_pressure = AoData(7);
current_pressure = min(current_pressure,2.5);
scaleData.Set(6,AoData(7));
errorCode = instantAoCtrl.Write(AOchannelStart, AOchannelCount, scaleData);
waitfor(Rate2);

% 恢复第二节
for i =step:-1:1
    current_pressure = AoData(5)/step*i;
    current_pressure = min(current_pressure,6);
    scaleData.Set(4,current_pressure);
    errorCode = instantAoCtrl.Write(AOchannelStart, AOchannelCount, scaleData);
    waitfor(Rate);
end

% 恢复第三节
scaleData.Set(6,0);
errorCode = instantAoCtrl.Write(AOchannelStart, AOchannelCount, scaleData);

waitfor(Rate2);

% 恢复第一节
for i =step:-1:1
    current_pressure = AoData(1)/step*i;
    current_pressure = min(current_pressure,5);
    scaleData.Set(0,current_pressure);
    errorCode = instantAoCtrl.Write(AOchannelStart, AOchannelCount, scaleData);
    waitfor(Rate);
end

