function [RMSE, Y_true, Y_pred] = evaluate_lko_poly(net, control, state, label, FeatDim, DelayTime)
    % 获取参数
    predict_step = size(control, 2);
    predict_step = 10;
    state_size = size(state,1);
    
    % 初始化
    Y_pred = zeros(state_size, predict_step);
    currentState = dlarray(state(:, 1), 'CB');
    
    for i=1:predict_step
        currentControl = dlarray(control(:, i), 'CB');    % 获取控制输入
        currentPolyFeat = dlarray(polynomial_expansion_td(stripdims(currentState), FeatDim, DelayTime), 'CB');    % 获取多项式特征
        predState = forward(net, currentState, currentPolyFeat, currentControl, 'Outputs', 'decoder');    % 获取预测
        
        Y_pred(:,i) = predState;
        currentState = predState;
    end

    Y_true = label(:, 1:predict_step);
    RMSE = calculateRMSE(Y_pred, Y_true);
end