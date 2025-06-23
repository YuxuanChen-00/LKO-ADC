function control_input = get_adaptive_control_input(e, dx_d, x_koopman_state, last_control_input, A, B, gamma, u_max, u_min, delta_u_max)
    control_input = pinv(B)*(gamma*e + dx_d - A*x_koopman_state);

end