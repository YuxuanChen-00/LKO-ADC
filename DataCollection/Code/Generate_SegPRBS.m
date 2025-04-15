function Generate_SegPRBS(D,N,f0,f1,seg_num,min_pressure,max_pressure)
    
    linearStep = 10;
    for k = 1:seg_num
        originalTime = 1:N/linearStep/seg_num;
        newTime = linspace(1, originalTime, N/seg_num);
        current_pressure = min_pressure + max_pressure;
        current_PRBS = Generate_PRBS(D,N/linearStep/seg_num,f0,f1,min_pressure,current_pressure);
        for d = 1:D
            current_PRBS(d, :) = interp1(originalTime, current_PRBS(d, :), newTime, 'linear');
        end
    end
    
    function y = Generate_PRBS(D,N,f0,f1,min_pressure,max_pressure)
        T = 1:1:N;
        y = zeros(D, N);
        for i = 1:D
            y(i,:) = idinput(length(T),'prbs',[f0 f1]);
            % 添加随机相位偏移
            y(i,:) = real(y(i,:).*exp(1i*rand*2*pi));
        end
        % 将信号缩放到指定范围内
        for j = 1:D
            current_min = min_pressure(j);
            current_max = max_pressure(j);
            y_min = min(y(j,:));
            y_max = max(y(j,:));
            y(j,:) = current_min + (y(j,:)-y_min)/(y_max-y_min)...
                *(current_max-current_min);
        end
    end
end

