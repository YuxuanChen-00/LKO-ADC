function delta_K = update_K(e, current_koopman_state, phi_error_window, phi_window, gamma1, gamma2)
    delta_K = e*current_koopman_state';
    for i = 1:phi_error_window.data_num
        delta_K = delta_K + gamma2*phi_error_window.history(i)*phi_window.hisotry(i)'/phi_error_window.data_num;
    end
    delta_K = gamma1 * delta_K;
end

