function total_loss = lstm_loss_function3(net, state, control, label, L1, L2, L3)
    pred_step = size(label, 2);
    batch_size = size(control, 3);
    state_size = size(state, 1);
    delay_step = size(state, 3);

    phi_pred_list = [];
    phi_true_list = [];
    state_pred_list = [];
    state_true_list = [];
    

    current_state_pred = state;
    for i = 1:pred_step
        current_control = dlarray(reshape(control(:,i,:),[],batch_size), 'CB');
        current_Phi_pred = forward(net, current_state_pred, current_control);  % 获取网络输出
        current_state_pred = current_Phi_pred(1:state_size*delay_step, :);
        current_next_Phi = forward(net, dlarray(squeeze(label(:,i,:,:)), 'CBT'), current_control,'Outputs', 'concat');
        current_next_state = current_next_Phi(1:state_size*delay_step, :);

        phi_pred_list = [phi_pred_list, current_Phi_pred];
        phi_true_list = [phi_true_list, current_next_Phi];
        state_pred_list = [state_pred_list, current_state_pred];
        state_true_list = [state_true_list, current_next_state];
        


        current_state_pred = dlarray(reshape(current_Phi_pred(1:state_size*delay_step, :),state_size, batch_size, delay_step),'CBT');
    end

    % 计算损失
    weights = net.Learnables.Value;
    l2Reg = L3*sum(cellfun(@(w) sum(w.^2, 'all'), weights)); % 计算L2正则项 
    loss_state = calculateRMSE(state_pred_list, state_true_list);
    loss_phi = calculateRMSE(phi_pred_list, phi_true_list);

    total_loss = L1*loss_state/pred_step + L2*loss_phi/pred_step+ l2Reg;
end