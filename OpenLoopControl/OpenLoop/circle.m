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
step = 80;

start_AoData = [0,0,0,0,0,0,0];
end_AoData = [0,0,0,4,0,0,0];

 for i = 1:step
    current_AoData = start_AoData+...
        (end_AoData-start_AoData)/step*i;
    current_AoData = min(current_AoData,4);
    AoWrite(current_AoData,instantAoCtrl,scaleData,AOchannelStart,AOchannelCount);
    waitfor(Rate);
 end

start_AoData = [0,0,0,4,0,0,0];
end_AoData = [0,0,0,4,4,0,0];
 for i = 1:step
    current_AoData = start_AoData+...
        (end_AoData-start_AoData)/step*i;
    current_AoData = min(current_AoData,4);
    AoWrite(current_AoData,instantAoCtrl,scaleData,AOchannelStart,AOchannelCount);
    waitfor(Rate);
 end
 
start_AoData = [0,0,0,4,4,0,0];
end_AoData = [0,0,0,0,5,0,0];
 for i = 1:step
    current_AoData = start_AoData+...
        (end_AoData-start_AoData)/step*i;
    current_AoData = min(current_AoData,4);
    AoWrite(current_AoData,instantAoCtrl,scaleData,AOchannelStart,AOchannelCount);
    waitfor(Rate);
 end

start_AoData = [0,0,0,0,4,0,0];
end_AoData = [0,0,0,0,4,4,0];
 for i = 1:step
    current_AoData = start_AoData+...
        (end_AoData-start_AoData)/step*i;
    current_AoData = min(current_AoData,4);
    AoWrite(current_AoData,instantAoCtrl,scaleData,AOchannelStart,AOchannelCount);
    waitfor(Rate);
 end

start_AoData = [0,0,0,0,4,4,0];
end_AoData = [0,0,0,0,0,4,0];
 for i = 1:step
    current_AoData = start_AoData+...
        (end_AoData-start_AoData)/step*i;
    current_AoData = min(current_AoData,4);
    AoWrite(current_AoData,instantAoCtrl,scaleData,AOchannelStart,AOchannelCount);
    waitfor(Rate);
 end

start_AoData = [0,0,0,0,0,4,0];
end_AoData = [0,0,0,5,0,4,0];
 for i = 1:step
    current_AoData = start_AoData+...
        (end_AoData-start_AoData)/step*i;
    current_AoData = min(current_AoData,4);
    AoWrite(current_AoData,instantAoCtrl,scaleData,AOchannelStart,AOchannelCount);
    waitfor(Rate);
 end

start_AoData = [0,0,0,4,0,4,0];
end_AoData = [0,0,0,4,0,0,0];
 for i = 1:step
    current_AoData = start_AoData+...
        (end_AoData-start_AoData)/step*i;
    current_AoData = min(current_AoData,4);
    AoWrite(current_AoData,instantAoCtrl,scaleData,AOchannelStart,AOchannelCount);
    waitfor(Rate);
 end
 
start_AoData = [0,0,0,4,0,0,0];
end_AoData = [0,0,0,0,0,0,0];
 for i = 1:step
    current_AoData = start_AoData+...
        (end_AoData-start_AoData)/step*i;
    current_AoData = min(current_AoData,4);
    AoWrite(current_AoData,instantAoCtrl,scaleData,AOchannelStart,AOchannelCount);
    waitfor(Rate);
 end