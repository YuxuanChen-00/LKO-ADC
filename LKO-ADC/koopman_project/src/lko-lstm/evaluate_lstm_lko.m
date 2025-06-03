function [RMSE, Y_true, Y_pred] = evaluate_lstm_lko(net, control, state, label, delay_step)
    predict_step = size(control, 3);
    % predict_step = 20;
    state_size = size(state,1);
    Y_pred = zeros(state_size, size(label,2));
    current_state = dlarray(state(:, 1, :), "CBT");
    for i=1:predict_step
        current_control = dlarray(reshape(control(:,1,i),[],1), 'CB');
        current_phi_pred = forward(net, current_state, current_control);  % 获取网络输出
        current_state_pred = current_phi_pred(1:state_size*delay_step, :);
        
        Y_pred(:,i) = current_phi_pred(1:state_size);
        current_state = dlarray(reshape(current_state_pred,state_size, 1, delay_step),'CBT');
    end
    
    Y_true = squeeze(label(:,1,1:predict_step,1));

    RMSE = calculateRMSE(Y_pred, Y_true);
end