function y = Generate_PRBS(D,N,f0,f1,min_pressure,max_pressure)
    T = 1:1:N;
    y = zeros(D, N);
    for i = 1:D
        y(i,:) = idinput(length(T),'prbs',[f0 f1]);
        % 添加随机相位偏移
        y(i,:) = real(y(i,:).*exp(1i*rand*2*pi));
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

