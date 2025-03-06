function high_dim_state = embedding_polynomial(state, p)

    high_dim_state = zeros(p, size(state,2));
    for k = 1:size(state, 2)
        current_state = state(:,k);
        high_dim_state(:,k) = poly_liftfunction(current_state, p);
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
