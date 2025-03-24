function total_loss = gcn_loss_function(net, state, control, label, L1, L2, feature_size, node_size)
    pred_step = size(label, 3);
    
    % L2正则化
    weights = net.Learnables.Value;
    l2Reg = L2*sum(cellfun(@(w) sum(w.^2, 'all'), weights)); % 计算L2正则项
    
    % 计算损失
    loss_state = 0;
    Phi_pred = [];
    state_pred = [];
    Phi = []
    for i = 1:pred_step
        current_Phi_pred = forward(net, state, control(:,i,:));  % 获取网络输出
        current_state_pred = current_Phi_pred(1:feature_size*node_size, :);
        current_Phi = forward(net, label(:,:,i,:), control(:,i,:),'Outputs', 'concat');
        current_state = label(:,:,i,:);

        Phi_pred = [Phi_pred, current_Phi_pred];
        state_pred = [state_pred, current_state_pred]
        Phi = [Phi, current_Phi]

        loss_state = loss_state + L1*mse(current_state_pred, dlarray(reshape(label(:,:,i,:),[],1,)))
    end
    total_loss = loss_state + loss_phi + l2Reg;
end

