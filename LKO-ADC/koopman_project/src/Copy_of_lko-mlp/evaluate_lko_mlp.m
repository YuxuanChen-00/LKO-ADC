function [RMSE, Y_true, Y_pred] = evaluate_lko_mlp(net, control, state, label, FeatDim)
    % 获取参数
    predict_step = size(control, 2);
    predict_step = 50;
    state_size = size(state,1);
    delay_step = size(state, 3);

    % 初始化
    Y_pred = zeros(state_size, predict_step);
    currentState = dlarray(state(:, 1, :), 'CBT');
    place_holder = dlarray(zeros(FeatDim, 1, delay_step), 'CBT');
    
    for i=1:predict_step
        currentControl = dlarray(control(:, i), 'CB');    % 获取控制输入

        predFeat = forward(net, currentState, currentControl, place_holder, 'Outputs', 'pred');    % 获取预测
        predState = forward(net, currentState, currentControl, predFeat, 'Outputs', 'decoder_out');
  
        
        Y_pred(:,i) = predState(:,:,1);
        currentState = predState;
    end

    Y_true = label(:, 1:predict_step, 1);
    RMSE = calculateRMSE(Y_pred, Y_true);
end