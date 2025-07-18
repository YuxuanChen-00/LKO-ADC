%% ========================================================================
%                     �������ڸ��º��� (���޸�)
% =========================================================================
function [grad_A_avg, grad_B_avg] = update_AB_sliding_window(A_current, B_current, ...
                                                   X_window, U_window, X_next_window)
% ʹ�û��������ڵ����ݣ�ͨ���ݶ��½�������A��B������ݶ�
%
% ����:
%   A_current, B_current: ��ǰ��A, B�������ֵ
%   X_window:      ״̬��ʷ����, ά��Ϊ (n_states x window_size), ���� x_k
%   U_window:      ������ʷ����, ά��Ϊ (n_inputs x window_size), ���� u_k
%   X_next_window: ��ʵ��һ״̬��ʷ����, ά��Ϊ (n_states x window_size), ���� x_{k+1}
%
% ���:
%   grad_A_avg, grad_B_avg: �������A��B��ƽ���ݶ�

    % ���ڳ���
    N_window = size(X_window, 2);
    
    % ��ʼ���ݶ��ۼ���
    grad_A_sum = zeros(size(A_current));
    grad_B_sum = zeros(size(B_current));
    
    % ���������ڵ����ݵ����ۼ��ݶ�
    for i = 1:N_window
        % ��ȡ���ݵ� (x_i, u_i) ����ʵ����һ״̬ x_{i+1}
        x_i = X_window(:, i);
        u_i = U_window(:, i);
        x_i_plus_1_real = X_next_window(:, i); % <-- ʹ����ȷ����ʵ��һʱ��״̬

        % ʹ�õ�ǰģ�ͽ���Ԥ��
        x_i_plus_1_pred = A_current * x_i + B_current * u_i;
        
        % ����״̬Ԥ�����
        error_vec = x_i_plus_1_pred - x_i_plus_1_real; % ע�⣺�ݶ��Ƶ��е������ pred-real 
                                                       % ���Ը���ʱ�� A = A - eta * grad���� A = A - eta * (pred-real)*x'
                                                       % �������Ϊ real-pred, ���·����� A = A + eta*grad
        
        % ������ʧ������A��B���ݶ�: L = 1/2 * ||error||^2
        % dL/dA = dL/de * de/dA = error * x_i'
        grad_A_single = error_vec * x_i';
        
        % dL/dB = dL/de * de/dB = error * u_i'
        grad_B_single = error_vec * u_i';
        
        % �ۼ��ݶ�
        grad_A_sum = grad_A_sum + grad_A_single;
        grad_B_sum = grad_B_sum + grad_B_single;
    end
    
    % ���ݶ���ƽ����ʹѧϰ�ʲ����������ڴ��ڴ�С
    grad_A_avg = grad_A_sum / N_window;
    grad_B_avg = grad_B_sum / N_window;
end