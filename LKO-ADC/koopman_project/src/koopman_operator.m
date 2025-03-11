% 该函数用于计算Koopman算子
function [A, B] = koopman_operator(control, state_phi, label_phi)
    control_size = size(control, 1);
    target_dimensions = size(state_phi, 1);

    K = label_phi*pinv([state_phi;control]);
    A = K(:,1:target_dimensions);
    B = K(:,target_dimensions+1:target_dimensions + control_size);
end

