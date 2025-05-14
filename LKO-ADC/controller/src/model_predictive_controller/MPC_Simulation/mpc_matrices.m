% 用于计算MPC控制所需矩阵
function  [E1,E2,H]=mpc_matrices(A,B,C,Q,R,F,N)
    n_state=size(A,1);   % A  n_state x n_state
    n_input=size(B,2);   % B  n_state x n_input 
    n_output=size(C,1);  % C  n_output x n_state

    M=[C;zeros(N*n_output,n_state)]; % 初始化 M 矩阵. M 矩阵是 (N+1)n x n的， 
                             % 它上面是 n x n 个 "I", 这一步先把下半部
                             % 分写成 0 
    D=zeros((N+1)*n_output,N*n_input); % 初始化 D 矩阵, 这一步令它有 (N+1)n x NP 个 0
    % 定义M 和 C 
    tmp=eye(n_state);  %定义一个n x n 的 I 矩阵
    %　更新Ｍ和C
    for i=1:N % 循环，i 从 1到 N
        rows =i*n_output+(1:n_output); %定义当前行数，从i x n开始，共n行 
        D(rows,:)=[C*tmp*B,D(rows-n_output, 1:end-n_input)]; %将D矩阵填满
        tmp= A*tmp; %每一次将tmp左乘一次A
        M(rows,:)=C*tmp; %将M矩阵写满
    end
    % 定义Q_bar和R_bar
    Q_bar = kron(eye(N),Q);
    Q_bar = blkdiag(Q_bar,F);
    R_bar = kron(eye(N),R);
    % 计算E, H
    E1=M'*Q_bar*D; 
    E2=Q_bar*D;
    H=D'*Q_bar*D+R_bar; 
end
