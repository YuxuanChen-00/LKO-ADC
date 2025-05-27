function high_dim_state = polynomial_expansion_td(state, p, delay_time)
    state_size = size(state,1)/delay_time;
    high_dim_state = zeros(p*delay_time, size(state,2));
    
    for k = 1:size(state, 2)
        for t = 1:delay_time
            index = state_size*(t-1)+1 : state_size*t;
            high_dim_index = p*(t-1)+1 : p*t;
            current_state = state(index, k);
            high_dim_state(high_dim_index,k) = adaptive_poly_lift(current_state, p);
        end
    end
end

function lifted = adaptive_poly_lift(x, target_dim)
    m = length(x);
    if target_dim < m
        error('目标维度需≥原状态维度');
    end
    
    lifted = x;  % 初始包含原始状态
    current_dim = m;
    max_order = 1;  % 初始最大阶数
    
    % 动态调节最高阶数
    while current_dim < target_dim
        max_order = max_order + 1;
        new_terms = generate_higher_order_terms(x, max_order);
        
        for i = 1:length(new_terms)
            if current_dim >= target_dim
                break;
            end
            lifted = [lifted; new_terms(i)];
            current_dim = current_dim + 1;
        end
        
        if max_order > 5  % 安全阈值
            warning('已达五阶仍未满足维度');
            break;
        end
    end
    
    % 维度校验
    if length(lifted) > target_dim
        lifted = lifted(1:target_dim);
    end
end

function terms = generate_higher_order_terms(x, order)
    terms = [];
    m = length(x);
    
    % 生成唯一组合项（避免重复）
    indices = nchoosek(1:m, order);
    unique_combs = unique(sort(indices,2), 'rows');
    
    for i = 1:size(unique_combs,1)
        term = prod(x(unique_combs(i,:)));
        terms = [terms; term];
    end
end