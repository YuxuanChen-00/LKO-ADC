function linearPressureControl(P_end, P_start, rate, instantAoCtrl_1,scaleData,AOchannelStart, AOchannelCount)
    % 线性变化的增量
    P_delta = (P_end - P_start) / 4; % 计算每次更新的增量
    % 通过rate控制每次气压更新
    for t = 1:4
        % 当前时刻的气压值
        P_current = P_start + P_delta * t;
        
        % 在此时刻设置气压，调用气动控制函数
        AoWrite(P_current,instantAoCtrl_1,scaleData,AOchannelStart, AOchannelCount)
        
        % 等待下一个周期
        waitfor(rate);
    end
end