function filtered_value = median_filter(current_value, last_value, current_window, threshold)
    D = size(current_window, 1);
    
    % 计算窗口内的中位数和MAD（中位数的绝对偏差）
    median_value = median(current_window,2);
    
    % 计算当前点与窗口中位数的偏差
    deviation = abs(current_value - median_value);
    % 异常值检测
    filtered_value = zeros(D,1);
    for i = 1:D
        if deviation(i) > threshold(i)
            filtered_value(i) = last_value(i);
        else
            filtered_value(i) = current_value(i);
        end
    end
    % disp([size(current_value),size(last_value),size(filtered_value)])
end
