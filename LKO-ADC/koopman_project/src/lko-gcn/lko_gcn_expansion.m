function high_dim_state = lko_gcn_expansion(state, net)
    high_dim_state = [];
    control_placeholder = dlarray(rand(6,1),'CB');
    for k = 1:size(state, 4)
        current_state = dlarray(state(:,:,:,k),'SSCB');

        disp(size(current_state));
        disp(size(control_placeholder))
        current_high_dim_state = forward(net, current_state, control_placeholder, 'Outputs', 'concat');
        high_dim_state = [high_dim_state, current_high_dim_state];
    end
end

