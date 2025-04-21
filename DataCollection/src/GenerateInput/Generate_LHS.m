function y = Generate_LHS(D,N,T,min_pressure,max_pressure)
    % 生成拉丁超立方样本
    X = lhsdesign(T*N,D);

    % 初始化气压输入信号矩阵（7通道）
    y = zeros(D, N);

    for d = 1:D
        % 对每个通道的信号进行线性插值以生成平滑过渡
        y(d, :) = interp1(linspace(0, 1, T*N), X(:, d), linspace(0, 1, N), 'spline');
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



