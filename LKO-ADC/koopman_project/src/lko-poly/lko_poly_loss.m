function [total_loss, polyWeight] = lko_poly_loss(net, state, control, label, polyStateFeat, polyLabelFeat, L1, L2, L3)
   
    % 获取预测的高维特征
    MixedPredFeat = forward(net, state, polyStateFeat, control, polyStateFeat, 'Outputs', 'add2');
    polyWeight = forward(net, state, polyStateFeat, control, polyStateFeat, 'Outputs', 'polyWeight');
    
    % 获取下一时刻高维特征
    MixedNextFeat = forward(net, label, polyLabelFeat, control, polyStateFeat, 'Outputs', 'add1');

    % 获取下一时刻高维特征的解码
    DecodeNextState = forward(net, state, polyStateFeat, control, MixedNextFeat, 'Outputs', 'decoder_out');

    % 特征预测损失
    loss_pred = L1*calculateRMSE(MixedPredFeat, MixedNextFeat);

    % 解码器损失
    loss_decoder = L2*calculateRMSE(label, DecodeNextState);


    % L2正则化
    weights = net.Learnables.Value;
    l2Reg = L3*sum(cellfun(@(w) sum(w.^2, 'all'), weights)); % 计算L2正则项
    
    % 损失相加
    total_loss = loss_pred + loss_decoder + l2Reg;
end