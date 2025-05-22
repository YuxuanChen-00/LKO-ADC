function high_dim_state = polynomial_expansion_td(state, p, delay_time)
    state_size = size(state,1)/delay_time;
    
    high_dim_state = zeros(p*delay_time, size(state,2));
    for k = 1:size(state, 2)
        for t = 1:delay_time
            index = state_size*(t-1) + 1 : state_size*t;
            high_dim_index = p*(t-1) + 1 : p*t;
            current_state = state(index, k);
            high_dim_state(high_dim_index,k) = poly_liftfunction(current_state, p);
        end
    end

    function high_dim_state_vector = poly_liftfunction(state_vector, target_dimensions)
        m = length(state_vector);
        if target_dimensions < m
            disp('高维状态的维度不能低于原状态');
            return;
        end
        % 创建一个存储高维状态的空向量
        high_dim_state_vector = state_vector;
        
        % 当前状态和导数乘积的计数
        count = 0;
        
        % 计算状态之间的乘积
        for i = 1:m
            for j = i:m
                if count < target_dimensions-m
                    high_dim_state_vector = [high_dim_state_vector; state_vector(i) * state_vector(j)];
                    count = count + 1;
                else
                    return;
                end
            end
        end
    end

end
