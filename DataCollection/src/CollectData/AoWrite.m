function AoWrite(AoData,instantAoCtrl_1,scaleData,AOchannelStart, AOchannelCount)
    %%模拟输出
    scaleData.Set(0,AoData(1));%将AoData转成AO0+data固定格式
    errorCode = instantAoCtrl_1.Write(AOchannelStart, AOchannelCount, scaleData);%data写到PCI1727#0 AO_1�?
    scaleData.Set(1,AoData(2));
    errorCode = instantAoCtrl_1.Write(AOchannelStart, AOchannelCount, scaleData);%data写到PCI1727#0 AO_2�?
    scaleData.Set(2,AoData(3));
    errorCode = instantAoCtrl_1.Write(AOchannelStart, AOchannelCount, scaleData);%data写到PCI1727#1 AO_3�?
    scaleData.Set(3,AoData(4));
    errorCode = instantAoCtrl_1.Write(AOchannelStart, AOchannelCount, scaleData);
    scaleData.Set(4,AoData(5));
    errorCode = instantAoCtrl_1.Write(AOchannelStart, AOchannelCount, scaleData);
    scaleData.Set(5,AoData(6));
    errorCode = instantAoCtrl_1.Write(AOchannelStart, AOchannelCount, scaleData);
    scaleData.Set(6,AoData(7)); 
    errorCode = instantAoCtrl_1.Write(AOchannelStart, AOchannelCount, scaleData);
end