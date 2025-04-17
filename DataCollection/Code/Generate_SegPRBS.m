function PRBS_signal = Generate_SegPRBS(D, N, f0, f1, seg_num, min_pressure, max_pressure)
    PRBS_signal = [];
    linearStep = 5; % 减小插值步长增强平滑度
    transition_len = 50; % 段间过渡长度
    
    for k = 1:seg_num
        current_max = min_pressure + (max_pressure - min_pressure) * k/seg_num;
        
        % 生成基础PRBS信号
        base_PRBS = Generate_PRBS(D, N/(linearStep*seg_num), f0, f1, min_pressure, current_max);
        
        % 使用三次样条插值
        originalTime = linspace(1, N/seg_num, size(base_PRBS,2));
        newTime = linspace(1, N/seg_num, N/seg_num);
        current_signal = zeros(D, N/seg_num);
        for d = 1:D
            current_signal(d,:) = interp1(originalTime, base_PRBS(d,:), newTime, 'pchip');
        end
        
        % 添加段间过渡
        if ~isempty(PRBS_signal)
            transition_zone = PRBS_signal(:,end-transition_len+1:end) .* linspace(1,0,transition_len)...
                            + current_signal(:,1:transition_len) .* linspace(0,1,transition_len);
            PRBS_signal = [PRBS_signal(:,1:end-transition_len), transition_zone, current_signal(:,transition_len+1:end)];
        else
            PRBS_signal = current_signal;
        end
    end
    function y = Generate_PRBS(D, N, f0, f1, min_pressure, max_pressure)
    % 新增参数
    smooth_window = 5; % 滑动平均窗长
    
    T = 1:N;
    y = zeros(D, N);
    for i = 1:D
        prbs = idinput(N, 'prbs', [f0 f1])';
        prbs = filter(ones(1,smooth_window)/smooth_window, 1, prbs); % 添加预平滑
        
        % 保持数值稳定性
        prbs = prbs - mean(prbs);
        
        % 动态范围调整（防止溢出）
        prbs = prbs / max(abs(prbs)) * 0.8; 
        
        % 相位随机化与缩放
        delay = randi(N);  
        y(i,:) = circshift(prbs, delay);
        y(i,:) = min_pressure(i) + (y(i,:)+1)/2 * (max_pressure(i)-min_pressure(i));
    end
end
end