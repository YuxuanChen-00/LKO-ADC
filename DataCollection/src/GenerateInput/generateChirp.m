function y = generateChirp(D, N, f0, f1, min_pressure, max_pressure)
    T = 1:N; % 时间序列从1到N
    y = zeros(D, N);
    
    for i = 1:D
        % 生成0到360度之间的随机相位
        phi = 360 * rand();
        % 生成带有随机相位的线性扫频信号
        y(i,:) = chirp(T, f0, T(end), f1, 'linear', phi);
    end
    
    % 将信号缩放到指定压力范围
    for j = 1:D
        current_min = min_pressure(j);
        current_max = max_pressure(j);
        % 计算当前维度的信号极值
        y_min = min(y(j,:));
        y_max = max(y(j,:));
        % 线性缩放
        y(j,:) = current_min + (y(j,:) - y_min) / (y_max - y_min) * (current_max - current_min);
    end
end