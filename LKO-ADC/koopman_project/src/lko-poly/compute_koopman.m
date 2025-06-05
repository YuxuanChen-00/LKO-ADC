function [A, B] = compute_koopman(U, X_phi, Y_phi, lambda)
    disp(size(X_phi));
    disp(size(U));
    Psi = [X_phi; U];
    reg_matrix = 0*lambda * eye(size(Psi,1));
    AB = Y_phi * Psi' / (Psi * Psi' + reg_matrix);
    
    state_dim = size(X_phi,1);
    A = AB(:, 1:state_dim);
    B = AB(:, state_dim+1:end);
end