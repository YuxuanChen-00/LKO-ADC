function total_loss = gcn_loss_function2(net, state, control, label, L1, L2, L3, feature_size, node_size)
    pred_step = size(label, 3);
    batch_size = size(control, 4);
    
    % L2正则化
    weights = net.Learnables.Value;
    l2Reg = L3*sum(cellfun(@(w) sum(w.^2, 'all'), weights)); % 计算L2正则项
    
    % 计算损失
    loss_state = 0;
    loss_phi = 0;
    current_state_pred = state;
    for i = 1:pred_step
        current_control = dlarray(reshape(control(:,:,i,:),[],batch_size), 'CB');
        current_Phi_pred = forward(net, current_state_pred, current_control, 'Outputs','concat');
        current_state_pred = dlarray(current_Phi_pred(1:feature_size*node_size, :), 'CB');
        


        current_next_state = dlarray(reshape(label(:,:,i,:),[],batch_size));
        current_next_phi = forward(net, label(:,:,i,:), current_control, 'Outputs','concat');

        loss_state = loss_state + L1*mse(current_state_pred(25:36,:), current_next_state(25:36,:));
        loss_phi = loss_phi + L2*mse(current_Phi_pred, current_next_phi);

        current_state_pred = dlarray(reshape(current_state_pred,6,6,1,batch_size),'SSCB');

    end
    total_loss = loss_state/pred_step + loss_phi/pred_step + l2Reg;
end

