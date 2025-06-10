function total_loss = lko_poly_loss(net, state, control, label, polyStateFeat, polyLabelFeat, L1, L2, L3)
    state_size = size(state, 1);
    control = stripdims(control);

    % 获取当前的高维特征
    currentFeat = stripdims(forward(net, state, polyStateFeat));
    
    % 获取下一时刻高维特征
    nextFeat = stripdims(forward(net, label, polyLabelFeat));

    % 计算Koopman算子
    [A ,B] = compute_koopman(control, currentFeat, nextFeat);
    predFeat = A*currentFeat + B*control;
    predState = predFeat(1:state_size, :);

    % 特征预测损失
    loss_state = calculateRMSE(predState, label);
    
    % 解码器损失
    loss_phi = calculateRMSE(predFeat, nextFeat);

    % L2正则化
    weights = net.Learnables.Value;
    l2Reg = sum(cellfun(@(w) sum(w.^2, 'all'), weights)); % 计算L2正则项

    % 损失相加
    total_loss = L1*loss_state + L2*loss_phi + L3*l2Reg;

    fprintf('状态损失: %.4f, 特征损失: %.4f\n', loss_state, loss_phi);
end