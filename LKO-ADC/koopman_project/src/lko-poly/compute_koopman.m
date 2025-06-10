function [A, B] = compute_koopman(U, X_phi, Y_phi)
    Psi = [X_phi; U];
    AB = Y_phi * pinv(Psi);
    
    state_dim = size(X_phi,1);
    A = AB(:, 1:state_dim);
    B = AB(:, state_dim+1:end);
end