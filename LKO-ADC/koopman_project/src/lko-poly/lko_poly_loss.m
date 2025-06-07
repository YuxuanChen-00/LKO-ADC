function total_loss = lko_poly_loss(net, state, control, label, polyStateFeat, polyLabelFeat, L1, L2, L3)
    
    % 获取当前的高维特征和预测的高维特征
    [predState, predFeat] = forward(net, state, polyStateFeat, control, 'Outputs', {'decoder', 'add2'});
    

    % 获取预测高维特征的解码和下一时刻高维特征
    nextFeat = forward(net, label, polyLabelFeat, control, 'Outputs', 'add2');

    % 特征预测损失
    loss_state = L1*calculateRMSE(predState, label);

    % 解码器损失
    loss_phi = L2*calculateRMSE(predFeat, nextFeat);

    % L2正则化
    weights = net.Learnables.Value;
    l2Reg = L3*sum(cellfun(@(w) sum(w.^2, 'all'), weights)); % 计算L2正则项

    % 损失相加
    total_loss = loss_state + loss_phi + l2Reg;

    fprintf('状态损失: %.4f, 特征损失: %.4f\n', loss_state/L1, loss_phi/L2);
end