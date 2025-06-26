function rmse = calculateRMSE(a, b)
    % 确保输入元素数目相同
    assert(numel(a) == numel(b), '输入的元素数目必须相同。');
    
    % 将输入转换为列向量（展平处理）
    a = a(:);
    b = b(:);
    
    % 计算RMSE
    diff = a - b;
    squared_diff = diff.^2;
    mse = mean(squared_diff);
    rmse = sqrt(mse);
end