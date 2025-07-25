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

samplerate=0.2;
Rate = robotics.Rate(samplerate);

% waitfor(Rate);


% �����Ӧ[0,6,6,4.5,0,4.5,0]
AoData=[0,0,0,0,0,0,3];
scaleData.Set(0,AoData(1));
errorCode = instantAoCtrl.Write(AOchannelStart, AOchannelCount, scaleData);
scaleData.Set(1,AoData(2));
errorCode = instantAoCtrl.Write(AOchannelStart, AOchannelCount, scaleData);
scaleData.Set(2,AoData(3));
errorCode = instantAoCtrl.Write(AOchannelStart, AOchannelCount, scaleData);
scaleData.Set(3,AoData(4));
errorCode = instantAoCtrl.Write(AOchannelStart, AOchannelCount, scaleData);
scaleData.Set(4,AoData(5));
errorCode = instantAoCtrl.Write(AOchannelStart, AOchannelCount, scaleData);
scaleData.Set(5,AoData(6));
errorCode = instantAoCtrl.Write(AOchannelStart, AOchannelCount, scaleData);
scaleData.Set(6,AoData(7));
errorCode = instantAoCtrl.Write(AOchannelStart, AOchannelCount, scaleData);

% waitfor(Rate);
% index = [1,2,3,7,8,9];
% [initRotationMatrix, initPosition] = getInitState(onemotion_data);
% last_sample = transferVicon2Base(onemotion_data, initRotationMatrix, initPosition);
% disp(last_sample(index))
