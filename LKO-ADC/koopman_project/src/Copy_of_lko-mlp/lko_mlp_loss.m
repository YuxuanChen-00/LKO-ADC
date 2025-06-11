function total_loss = lko_mlp_loss(net, state, control, label, L1, L2, L3, FeatDim)
    % 获取部分参数
    batch_size = size(control, 2);
    delay_time = size(state, 3);
    place_holder = dlarray(zeros(FeatDim, batch_size, delay_time), 'CBT');

    % 获取当前的高维特征和预测的高维特征
    [currentFeat, predFeat] = forward(net, state,  control, place_holder, 'Outputs', {'encoder_out', 'pred'}); 
    
    % 获取当前时刻高维特征解码和下一时刻特征
    [currentDecode, nextFeat] = forward(net, label,  control, currentFeat, 'Outputs', {'decoder_out', 'encoder_out'}); 

    % 获取下一时刻高维特征的解码
    nextDecode = forward(net, label,  control, nextFeat, 'Outputs', 'decoder_out'); 
    predDecode = forward(net, label, control, predFeat, 'Outputs', 'decoder_out');

    % 获取整形之后的state

    % 计算损失
    loss_decoder = L2*calculateRMSE(currentDecode, state) + L2*calculateRMSE(nextDecode, label) + L2*calculateRMSE(predDecode, label);
    loss_pred = L1*calculateRMSE(predFeat, nextFeat);
    weights = net.Learnables.Value;
    l2Reg = L3*sum(cellfun(@(w) sum(w.^2, 'all'), weights)); % 计算L2正则项

    total_loss = loss_decoder + loss_pred+ l2Reg;

    fprintf('解码器损失: %.4f, 预测损失: %.4f\n', loss_decoder/L2, loss_pred/L1);
end