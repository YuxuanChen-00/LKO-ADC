function pred_phi = predict_multistep(A, B, control, state_phi, step)
    pred_phi = zeros(size(state_phi, 1), step);
    current_state_phi = state_phi;
    for i = 1:1:step
        pred_phi(:, i) = predict_singelstep(A, B, control(:, i), current_state_phi);
        current_state_phi = pred_phi(:, i);
    end
end

