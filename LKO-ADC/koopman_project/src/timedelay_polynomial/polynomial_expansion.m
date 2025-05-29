function high_dim_state = polynomial_expansion(state, p, delay_time)
    % 参数校验
    if mod(size(state,1), delay_time) ~= 0
        error('状态维度必须能被延迟时间整除');
    end
    
    state_size = size(state,1) / delay_time;
    high_dim_state = zeros(p * delay_time, size(state,2));
    
    % 时间延迟嵌入处理
    for k = 1:size(state, 2)
        for t = 1:delay_time
            % 提取当前延迟时段的状态
            idx = (t-1)*state_size + 1 : t*state_size;
            current_state = state(idx, k);
            
            % 自适应升维
            hd_idx = (t-1)*p + 1 : t*p;
            high_dim_state(hd_idx, k) = adaptive_poly_lift(current_state, p);
        end
    end
end

function lifted = adaptive_poly_lift(x, target_dim)
    m = length(x);
    if target_dim < m
        error('目标维度必须≥原状态维度');
    end
    
    % 初始化存储空间
    lifted = zeros(target_dim, 1);
    lifted(1:m) = x(:);
    current_dim = m;
    current_order = 2;  % 从二阶开始
    
    % 生成各阶项直到满足维度
    while current_dim < target_dim && current_order <= 5
        % 生成唯一组合索引
        comb = nchoosek(1:m, current_order);
        comb = unique(sort(comb,2), 'rows', 'stable');
        
        % 动态填充高阶项
        for i = 1:size(comb,1)
            if current_dim >= target_dim
                break;
            end
            current_dim = current_dim + 1;
            lifted(current_dim) = prod(x(comb(i,:)));
        end
        
        current_order = current_order + 1;
    end
end