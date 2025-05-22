function [A, B] = koopman_operator(control, state_phi, label_phi)
    % 参数校验
    
    x_percent = 0;

    if x_percent < 0 || x_percent >= 100
        error('截断比例x_percent需在[0,100)区间');
    end
    
    control_size = size(control, 1);
    target_dimensions = size(state_phi, 1);
    
    % 构建增广矩阵
    M = [state_phi; control];
    
    % 经济型SVD分解[1,8](@ref)
    [U, S, V] = svd(M, 'econ');
    s = diag(S);  % 获取奇异值向量
    
    if x_percent > 0
        % 计算奇异值能量累积比例[6,7](@ref)
        total_energy = sum(s.^2);
        cum_energy = cumsum(s.^2) / total_energy;
        
        % 确定保留阈值[8](@ref)
        k = find(cum_energy >= (1 - x_percent/100), 1, 'first');
        if isempty(k)
            k = length(s);
        end
        
        % 截断奇异值[6,8](@ref)
        s = s(1:k);
        U = U(:,1:k);
        V = V(:,1:k);
    end
    
    % 构建截断伪逆矩阵[8](@ref)
    S_inv = diag(1./s);
    M_pinv = V * S_inv * U';
    
    % 计算Koopman算子[1,4](@ref)
    K = label_phi * M_pinv;
    
    % 分解动态矩阵A和控制矩阵B[4](@ref)
    A = K(:,1:target_dimensions);
    B = K(:,target_dimensions+1:target_dimensions + control_size);
end