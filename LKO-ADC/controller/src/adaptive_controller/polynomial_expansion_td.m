function high_dim_state = polynomial_expansion_td(state, p, delay_time)
    state_size = size(state,1)/delay_time;
    high_dim_state = zeros(p*delay_time, size(state,2));
   
    for k = 1:size(state, 2)
        for t = 1:delay_time
            index_range = state_size*(t-1)+1 : state_size*t;
            hd_index = p*(t-1)+1 : p*t;
            current_state = state(index_range, k);
            high_dim_state(hd_index,k) = adaptive_poly_lift(current_state, p);
        end
    end
end

function lifted = adaptive_poly_lift(x, target_dim)
    m = length(x);
    if target_dim < m
        error('Target dimension must ≥ original state dimension');
    end
    
    lifted = x(:);  % 初始状态列向量
    current_dim = m;
    current_order = 2;  % 从二阶开始
    
    % 生成各阶项直到满足维度
    while current_dim < target_dim && current_order <= 5
        % 生成唯一组合索引
        comb_indices = nchoosek(1:m, current_order);
        unique_combs = unique(sort(comb_indices,2), 'rows');
        
        % 按字典序遍历组合
        for i = 1:size(unique_combs,1)
            if current_dim >= target_dim
                break; 
            end
            
            % 动态计算多项式项
            term = prod(x(unique_combs(i,:)));
            lifted = [lifted; term];
            current_dim = current_dim + 1;
        end
        
        current_order = current_order + 1;
    end
    
    % 最终维度校验
    if length(lifted) > target_dim
        lifted = lifted(1:target_dim);
    end
end