function [RMSE, Y_true, Y_pred] = evaluate_lko_poly(net, control, state, label, FeatDim, DelayTime, A, B)
    % 获取参数
    % predict_step = size(control, 2);
    predict_step = 10;
    state_size = size(state,1);

    % 初始化
    Y_pred = zeros(state_size, predict_step);
    currentState = dlarray(state(:, 1), 'CB');

    % 获取多项式特征
    PolyFeat = polynomial_expansion_td(state, FeatDim, DelayTime);
    
    
    for i=1:predict_step
        currentControl = control(:, i);    % 获取控制输入
        currentPolyFeat = dlarray(PolyFeat(:, i), 'CB');    % 获取多项式特征
        currentMixFeat = forward(net, currentState, currentPolyFeat, currentPolyFeat, 'Outputs', 'gate_layer');    % 获取预测的高维特征
        predMixFeat = A*stripdims(currentMixFeat) + B*currentControl;
        predState = forward(net, currentState, currentPolyFeat, predMixFeat, 'Outputs', 'decoder_out');    % 获取解码后的原状态预测
        
        Y_pred(:,i) = predState;
        currentState = predState;
    end

    Y_true = label(:, 1:predict_step);
    RMSE = calculateRMSE(Y_pred, Y_true);
end