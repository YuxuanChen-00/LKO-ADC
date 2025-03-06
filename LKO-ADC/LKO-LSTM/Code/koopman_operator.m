% 该函数用于计算Koopman算子
function [A, B] = koopman_operator(control, state, label, lift_function, target_dimensions)

    state_phi = zeros(target_dimensions, size(state,2));
    label_phi = zeros(target_dimensions, size(state,2));
    for i = 1:size(state,2)
        state_phi(:, i) = lift_function(state(:, i), target_dimensions);
        label_phi(:,i) = lift_function(label(:,i), target_dimensions);
    end
    
    control_size = size(control, 1);
    
    K = label_phi*pinv([state_phi;control]);
    A = K(:,1:target_dimensions);
    B = K(:,target_dimensions+1:target_dimensions + control_size);

end

