function PRBS_signal = Generate_SegPRBS(D, N, f0, f1, seg_num, min_pressure, max_pressure)
    PRBS_signal = [];
    linearStep = 20;
    for k = 1:seg_num
        originalTime = 1:N/linearStep/seg_num;
        newTime = linspace(1, N/linearStep/seg_num, N/seg_num);
        current_pressure = min_pressure + k * max_pressure / seg_num;
        current_PRBS = Generate_PRBS(D, N/linearStep/seg_num, f0, f1, min_pressure, current_pressure);

        current_signal = zeros(D, N/seg_num);
        
        for d = 1:D
            current_signal(d,:) = interp1(originalTime, current_PRBS(d, :), newTime, 'linear');
        end

        PRBS_signal = [PRBS_signal, current_signal];
    end

function y = Generate_PRBS(D, N, f0, f1, min_pressure, max_pressure)
    T = 1:N;
    y = zeros(D, N);
    for i = 1:D
        % 生成PRBS信号
        prbs = idinput(N, 'prbs', [f0 f1])';
        
        % 添加随机循环平移（相位偏移）
        delay = randi(N);  % 生成1到N之间的随机整数
        y(i,:) = circshift(prbs, delay);  % 循环平移信号
        
        % 缩放信号到指定范围
        current_min = min_pressure(i);
        current_max = max_pressure(i);
        y_min = min(y(i,:));
        y_max = max(y(i,:));
        y(i,:) = current_min + (y(i,:) - y_min) / (y_max - y_min) * (current_max - current_min);
   
    end
end
end