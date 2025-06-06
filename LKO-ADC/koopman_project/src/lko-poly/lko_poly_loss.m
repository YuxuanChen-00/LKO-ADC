function total_loss = lko_poly_loss(net, state, control, label, polyStateFeat, polyLabelFeat, L1, L2, L3)
   
    % 获取当前时刻高维状态的解码和预测的高维特征
    [DecodePredState, MixedPredFeat] = forward(net, state, polyStateFeat, control, 'Outputs', {'decoder_out', 'add2'});
    
    % 获取下一时刻高维特征和下一时刻高维状态的解码
    MixedNextFeat = forward(net, label, polyLabelFeat, control, 'Outputs', 'mixLayer_out');

    % 特征预测损失
    loss_pred = L1*calculateRMSE(MixedPredFeat, MixedNextFeat);

    % 解码器损失
    loss_decoder = L2*calculateRMSE(label, DecodePredState);


    % L2正则化
    weights = net.Learnables.Value;
    l2Reg = L3*sum(cellfun(@(w) sum(w.^2, 'all'), weights)); % 计算L2正则项
    % 损失相加
    total_loss = loss_pred + loss_decoder + l2Reg;

    % fprintf('解码器损失: %.4f, 预测损失: %.4f\n', loss_decoder, loss_pred);

end