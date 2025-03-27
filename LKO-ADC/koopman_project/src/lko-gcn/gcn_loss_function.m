function total_loss = gcn_loss_function(net, state, control, label, L1, L2, L3, feature_size, node_size)
    pred_step = size(label, 3);
    batch_size = size(control, 4);

    % L2正则化
    weights = net.Learnables.Value;
    l2Reg = L3*sum(cellfun(@(w) sum(w.^2, 'all'), weights)); % 计算L2正则项
    
    % 计算损失
    loss_state = 0;
    loss_phi = 0;
    current_state_pred = state;
    current_control = dlarray(reshape(control(:,:,1,:),[],batch_size), 'CB');
    current_Phi_pred = extractdata(forward(net, current_state_pred, current_control, 'Outputs','concat'));
    A = net.Layers(8).Weights;
    B = net.Layers(9).Weights;

    for i = 1:pred_step
        current_control = dlarray(reshape(control(:,:,1,:),[],batch_size), 'CB');
        current_Phi_pred = A*current_Phi_pred+B*extractdata(current_control);
        current_state_pred = dlarray(current_Phi_pred(1:feature_size*node_size, :), 'CB');
        current_next_state = dlarray(reshape(label(:,:,i,:),[],batch_size));
        current_next_phi = forward(net, label(:,:,i,:), current_control, 'Outputs','concat');
        loss_state = loss_state + L1*mse(current_state_pred(25:36,:), current_next_state(25:36,:));
        loss_phi = loss_phi + L2*mse(dlarray(current_Phi_pred,'CB'), current_next_phi);
    end
    total_loss = loss_state/pred_step + loss_phi/pred_step + l2Reg;
end

