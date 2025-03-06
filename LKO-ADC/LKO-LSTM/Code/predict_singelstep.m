function pred_phi = predict_singelstep(A, B, control, state_phi)
    pred_phi = zeros(size(state_phi));
    for i = 1:size(state_phi, 2)
        pred_phi(:,i) = A*state_phi(:,i) + B*control(:,i);
    end
end

