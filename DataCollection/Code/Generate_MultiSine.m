function y = Generate_MultiSine(D, N, F, min_pressure, max_pressure)
    y = zeros(D, N);
    t = 1:1:N;
    for i = 1:D
        signal = zeros(1, N);
        
        for j = 1:size(F, 1)
            min_freq = F(j, 1);
            max_freq = F(j, 2);
            % 随机选择一个频率和相位
            freq = min_freq + (max_freq - min_freq) * rand();
            phase = 2 * pi * rand(); % 生成0到2π的随机相位
            % 生成带相位偏移的正弦波并叠加
            signal = signal + sin(2 * pi * freq * t + phase);
        end
        
        % 将生成的信号赋值到对应通道
        y(i, :) = signal;
    end
    
    % 将信号缩放到指定范围内
    for j = 1:D
        current_min = min_pressure(j);
        current_max = max_pressure(j);
        y_min = min(y(j, :));
        y_max = max(y(j, :));
        % 避免除以零（当y_max == y_min时，信号为常数值）
        if y_max ~= y_min
            y(j, :) = current_min + (y(j, :) - y_min) / (y_max - y_min) * (current_max - current_min);
        else
            y(j, :) = (current_min + current_max) / 2; % 设置为范围中点
        end
    end
end