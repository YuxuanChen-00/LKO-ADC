function y = Generate_RandomWalk(D, N, T, maxStep, probRise, probFall, probHold, min_pressure, max_pressure)
    if abs(probRise + probFall + probHold - 1) > 1e-6
        disp(abs(probRise + probFall + probHold - 1))
        error('probRise, probFall, probHold 的和必须为1');
    end
    
    size_r = N*T;
    random_data = zeros(D, size_r); % 初始化信号
    random_data(:,1) = (max_pressure-min_pressure).*rand(7,1) ; % 随机初始化第一个值

    for k = 2:size_r
        currentValue = random_data(:,k-1);
        
        % 根据概率选择状态
        r = rand(7,1);
        direction = zeros(7,1);
        for i=1:7
            if r(i) < probRise
                direction(i) = 1;
            elseif r(i) < probRise + probFall
                direction(i) = -1;
            end
        end
        % 计算步幅
        step = direction .* (rand(7,1) * maxStep);
        
        % 计算新值
        random_data(:,k) = currentValue + step;
        random_data(:,k) = max(min_pressure, min(max_pressure, random_data(:,k)));
    end
    
    % 对每个通道的信号进行插值以生成平滑过渡
    y = zeros(D, N);
    for d = 1:D
        y(d, :) = interp1(linspace(0, 1, T*N), random_data(d, :), linspace(0, 1, N), 'spline');
    end
end
