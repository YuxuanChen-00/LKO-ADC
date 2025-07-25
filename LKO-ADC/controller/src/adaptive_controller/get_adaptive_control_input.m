function control_input = get_adaptive_control_input(e, dx_d, x_koopman_state, last_control_input, A, B, K, u_max, u_min, delta_u_max)
    
    control_input = pinv(B)*(K*e + dx_d - A*x_koopman_state);

    delta_u = control_input - last_control_input;
    for i = 1:size(control_input, 1)
        if abs(delta_u(i)) > delta_u_max
            control_input(i) = last_control_input(i) + sign(delta_u(i))*delta_u_max;
        end
    end

    control_input = min(max(u_min, control_input), u_max);
end
