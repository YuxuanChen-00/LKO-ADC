function linearPressureControl(P_end, P_start, instantAoCtrl_1,scaleData,AOchannelStart, AOchannelCount)
    rate = robotics.Rate(100); % 创建控制速率对象
    
    % 线性变化的增量
    P_delta = (P_end - P_start) / 5; % 计算每次更新的增量
    % 通过rate控制每次气压更新
    for t = 1:5
        % 当前时刻的气压值
        P_current = P_start + P_delta * t;
        
        % 在此时刻设置气压，调用气动控制函数
        AoWrite(P_current,instantAoCtrl_1,scaleData,AOchannelStart, AOchannelCount)
        
        % 等待下一个周期
        waitfor(rate);
    end

function AoWrite(AoData,instantAoCtrl_1,scaleData,AOchannelStart, AOchannelCount)
    scaleData.Set(0,AoData(1));
    errorCode = instantAoCtrl_1.Write(AOchannelStart, AOchannelCount, scaleData);
    scaleData.Set(1,AoData(2));
    errorCode = instantAoCtrl_1.Write(AOchannelStart, AOchannelCount, scaleData);
    scaleData.Set(2,AoData(3));
    errorCode = instantAoCtrl_1.Write(AOchannelStart, AOchannelCount, scaleData);
    scaleData.Set(3,AoData(4));
    errorCode = instantAoCtrl_1.Write(AOchannelStart, AOchannelCount, scaleData);
    scaleData.Set(4,AoData(5));
    errorCode = instantAoCtrl_1.Write(AOchannelStart, AOchannelCount, scaleData);
    scaleData.Set(5,AoData(6));
    errorCode = instantAoCtrl_1.Write(AOchannelStart, AOchannelCount, scaleData);
    scaleData.Set(6,AoData(7)); 
    errorCode = instantAoCtrl_1.Write(AOchannelStart, AOchannelCount, scaleData);
end

end