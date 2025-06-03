function [RMSE, Y_true, Y_pred] = evaluate_lstm_lko2(net, control, state, label, delay_step)
    predict_step = size(control, 3);
    state_size = size(state,1);
    Y_pred = zeros(state_size, size(label,2));
    current_state = dlarray(state(:, 1, :), "CBT");
    current_control = dlarray(reshape(control(:,1,1),[],1), 'CB');
    current_phi = stripdims(forward(net, current_state, current_control));
    A = net.Layers(8).Weights;
    B = net.Layers(9).Weights;
    for i=1:predict_step
        current_control =reshape(control(:,1,i),[],1);
        current_phi = A*current_phi + B*current_control;
        Y_pred(:,i) = current_phi(1:state_size);
    end
    
    Y_true = squeeze(label(:,1,1:predict_step,1));

    RMSE = calculateRMSE(Y_pred, Y_true);
end