% 二次规划优化算法
function u_k= mpc_prediction(x_k,X_rk,E1,E2,H,p,A,b)
    F = x_k'*E1-X_rk'*E2;
    U_k = quadprog(H,F,A,b);
    u_k = U_k(1:p,1); % 取第一个结果
end 