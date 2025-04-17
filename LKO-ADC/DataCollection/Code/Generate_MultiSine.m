function y = Generate_MultiSine(D,N,F, min_pressure,max_pressure)
    y = zeros(D, N);
    t = 1:1:N;
    for i = 1:D
        signal = zeros(1,N);
        
        for j = 1:size(F,1)
             min_freq = F(j,1);
             max_freq = F(j,2);
             % 随机选择一个频率
             freq = min_freq + (max_freq-min_freq)*rand;
             % 生成该频率正弦波并叠加
             signal = signal + sin(2*pi*freq*t);
             
        end
        
        % 将生成的信号赋值到对应通道
        y(i,:) = signal;
    end
    
    % 将信号防缩到指定范围内
    for j = 1:D
        current_min = min_pressure(j);
        current_max = max_pressure(j);
        y_min = min(y(j,:));
        y_max = max(y(j,:));
        y(j,:) = current_min + (y(j,:)-y_min)/(y_max-y_min)...
            *(current_max-current_min);
    end
end

