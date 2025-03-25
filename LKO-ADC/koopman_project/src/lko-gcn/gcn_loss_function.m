function total_loss = gcn_loss_function(net, state, control, label, L1, L2, L3, feature_size, node_size)
    pred_step = size(label, 3);
    
    % L2正则化
    weights = net.Learnables.Value;
    l2Reg = L3*sum(cellfun(@(w) sum(w.^2, 'all'), weights)); % 计算L2正则项
    
    % 计算损失
    loss_state = 0;
    loss_phi = 0;
    % Phi_pred = [];
    % state_pred = [];
    % Phi = []
    current_state_pred = state;
    for i = 1:pred_step
        current_Phi_pred = forward(net, current_state_pred, dlarray(control(:,i,:),'CB'));  % 获取网络输出
        current_state_pred = current_Phi_pred(1:feature_size*node_size, :);
        current_next_Phi = forward(net, label(:,:,i,:), dlarray(control(:,i,:),'CB'),'Outputs', 'concat');
        current_next_state = label(:,:,i,:);
 

        % Phi_pred = [Phi_pred, current_Phi_pred];
        % state_pred = [state_pred, current_state_pred]
        % Phi = [Phi, current_Phi]
        disp(size(control))
        disp(size(label))
        disp(size(current_next_state))
        disp(size(current_state_pred))
        disp(size(dlarray(reshape(current_next_state,[],size(current_next_state,4)))))
        loss_state = loss_state + L1*mse(current_state_pred, dlarray(reshape(current_next_state,[],size(current_next_state,4))));
        loss_phi = loss_phi + L2*mse(current_next_Phi, current_Phi_pred);
    end
    total_loss = loss_state/pred_step + loss_phi/pred_step + l2Reg;
end

