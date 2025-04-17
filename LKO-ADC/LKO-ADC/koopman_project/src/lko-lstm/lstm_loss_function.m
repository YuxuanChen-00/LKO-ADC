function total_loss = lstm_loss_function(net, state, control, label, L1, L2, L3, state_size, time_step)
    pred_step = size(label, 3);
    batch_size = size(control, 2);
    disp(batch_size)
    % L2正则化
    weights = net.Learnables.Value;
    l2Reg = L3*sum(cellfun(@(w) sum(w.^2, 'all'), weights)); % 计算L2正则项
    
    % 计算损失
    loss_state = 0;
    loss_phi = 0;
    current_state_pred = state;
    for i = 1:pred_step
        current_control = dlarray(reshape(control(:,i,:),[],batch_size), 'CB');
        disp(size(current_control))
        disp(size(current_state_pred))
        current_Phi_pred = forward(net, current_state_pred, current_control);  % 获取网络输出
        current_state_pred = current_Phi_pred(1:state_size*time_step, :);
        current_next_Phi = forward(net, dlarray(squeeze(label(:,:,i,:)), 'CBT'), current_control,'Outputs', 'concat');
        current_next_state = label(:,:,i,:);


        loss_state = loss_state + L1*mse(current_state_pred, dlarray(reshape(current_next_state,[],batch_size)));
        loss_phi = loss_phi + L2*mse(current_next_Phi, current_Phi_pred);

        current_state_pred = dlarray(reshape(current_state_pred,state_size, batch_size, time_step),'CBT');
    end
    total_loss = loss_state/pred_step + loss_phi+ l2Reg;
end