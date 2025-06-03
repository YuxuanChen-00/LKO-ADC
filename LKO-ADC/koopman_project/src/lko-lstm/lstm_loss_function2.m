function total_loss = lstm_loss_function2(net, state, control, label, L1, L2, L3)
    pred_step = size(label, 2);
    batch_size = size(control, 3);
    state_size = size(state, 1);
    delay_step = size(state, 3);


    A = net.Layers(8).Weights;
    B = net.Layers(9).Weights;
    current_control = dlarray(reshape(control(:,1,:),[],batch_size), 'CB');
    current_phi_pred = stripdims(forward(net, state, current_control, 'Outputs', 'concat'));  % 获取网络输出


    phi_pred_list = [];
    phi_true_list = [];
    state_pred_list = [];
    state_true_list = [];

    for i = 1:pred_step
        current_control = dlarray(reshape(control(:,i,:),[],batch_size), 'CB');

        current_phi_pred = A*current_phi_pred + B*stripdims(current_control);
        current_phi_pred2 = stripdims(forward(net, state, current_control));
        save('phi.mat', "current_phi_pred", "current_phi_pred2")
        % disp(calculateRMSE(current_phi_pred2, current_phi_pred));


        current_state_pred = current_phi_pred(1:state_size*delay_step, :);

        current_phi_next = forward(net, dlarray(squeeze(label(:,i,:,:)), 'CBT'), current_control,'Outputs', 'concat');
        current_next_state = current_phi_next(1:state_size*delay_step, :);

        phi_pred_list = [phi_pred_list, current_phi_pred];
        phi_true_list = [phi_true_list, current_phi_next];
        state_pred_list = [state_pred_list, current_state_pred];
        state_true_list = [state_true_list, current_next_state];


    end
    % 计算损失
    loss_state = calculateRMSE(state_pred_list, state_true_list);
    loss_phi = calculateRMSE(phi_pred_list, phi_true_list);
    weights = net.Learnables.Value;
    l2Reg = sum(cellfun(@(w) sum(w.^2, 'all'), weights)); % 计算L2正则项

    total_loss = L1*loss_state/pred_step + L2*loss_phi/pred_step+ L3*l2Reg;
end