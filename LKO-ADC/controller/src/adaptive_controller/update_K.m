function delta_K = update_K(e, current_koopman_state, phi_error_window, phi_window, gamma1, gamma2)
    delta_K1 = e*current_koopman_state';
    delta_K2 = zeros(size(delta_K1));
    for i = 1:phi_error_window.data_num
        delta_K2 = delta_K2 + gamma2*phi_error_window.history(i)*phi_window.history(i)'/phi_error_window.data_num;
    end
    delta_K = gamma1 * (delta_K1 + delta_K2);
end

