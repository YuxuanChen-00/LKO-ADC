function filtered_signal = apply_sg_filter(signal_col_vector, poly_order, frame_len)
% apply_sg_filter: 手动实现Savitzky-Golay滤波器。
%
% 输入参数:
%   signal_col_vector - 输入信号向量 (单通道数据，必须为列向量)
%   poly_order        - Savitzky-Golay滤波器的多项式阶数
%   frame_len         - Savitzky-Golay滤波器的帧长度 (必须为奇数)
%
% 输出参数:
%   filtered_signal   - 滤波后的信号向量 (列向量)

    % --- 参数初步校验与调整 ---
    original_frame_len_for_msg = frame_len; 
    if mod(frame_len, 2) == 0 
        frame_len = frame_len + 1;
        warning_msg = sprintf('警告: 输入的帧长度 (frame_len = %d) 为偶数，已自动调整为奇数 %d。', original_frame_len_for_msg, frame_len);
        disp(warning_msg);
    end
    if poly_order < 0
        error('错误: 多项式阶数 (poly_order) 不能为负数。');
    end
    if frame_len <= 0 
         error('错误: 帧长度 (frame_len) 必须为正奇数。');
    end
    if poly_order >= frame_len && frame_len > 1 
        error('错误: 多项式阶数 (poly_order = %d) 必须小于帧长度 (frame_len = %d)。', poly_order, frame_len);
    end
    if frame_len == 1
        if poly_order > 0
            error('错误: 当帧长度为1时，多项式阶数必须为0。');
        else 
            filtered_signal = signal_col_vector; % 帧长为1，0阶多项式，不改变信号
            return;
        end
    end

    n = length(signal_col_vector); 
    filtered_signal = signal_col_vector; % 初始化为原始信号

    % --- 处理信号长度相对于滤波器参数的情况 ---
    effective_poly_order = poly_order; 
    effective_frame_len = frame_len;   

    if n < effective_poly_order + 1 
        warning_msg = sprintf('警告: 信号长度 (%d) 小于 (多项式阶数+1 = %d)。不进行滤波，返回原始信号。', n, effective_poly_order+1);
        disp(warning_msg);
        return;
    end

    if effective_frame_len > n
        original_frame_len_for_msg = effective_frame_len; 
        effective_frame_len = n; 
        if mod(effective_frame_len, 2) == 0 
            if effective_frame_len > 1
                effective_frame_len = effective_frame_len - 1;
            else 
                 effective_frame_len = 1; 
            end
        end
        
        if effective_poly_order >= effective_frame_len
            effective_poly_order = effective_frame_len - 1;
        end
        if effective_poly_order < 0 
            effective_poly_order = 0;
        end
        warning_msg = sprintf(['警告: 指定帧长 (%d) 大于信号长度 (%d)。\n', ...
            '实际使用帧长调整为 %d，多项式阶数调整为 %d。'], ...
            original_frame_len_for_msg, n, effective_frame_len, effective_poly_order);
        disp(warning_msg);
    end
    
    if effective_frame_len <= 0
        warning('警告: 计算得到的有效帧长 (%d) 无效。不进行滤波，返回原始信号。', effective_frame_len);
        return;
    end
    if effective_frame_len == 1 % 如果有效帧长最终为1
        if effective_poly_order > 0, effective_poly_order = 0; end % 阶数也必须为0
    end


    half_win = (effective_frame_len - 1) / 2; 

    % --- 对信号中的每个点进行滤波 ---
    for i = 1:n 
        start_idx_in_signal = max(1, i - half_win);
        end_idx_in_signal = min(n, i + half_win);
        
        current_window_data = signal_col_vector(start_idx_in_signal : end_idx_in_signal);
        current_window_len = length(current_window_data);
        
        t_local = (start_idx_in_signal : end_idx_in_signal)' - i;
        
        actual_poly_order_for_window = effective_poly_order;
        if current_window_len <= actual_poly_order_for_window
            actual_poly_order_for_window = current_window_len - 1;
        end
        if actual_poly_order_for_window < 0 
            actual_poly_order_for_window = 0;
        end

        X_vand = zeros(current_window_len, actual_poly_order_for_window + 1);
        for k = 0:actual_poly_order_for_window
            X_vand(:, k+1) = t_local.^k;
        end
        
        if rank(X_vand) < (actual_poly_order_for_window + 1)
            if ~isempty(current_window_data)
                filtered_signal(i) = mean(current_window_data);
            end
        else
            poly_coeffs = X_vand \ current_window_data; 
            filtered_signal(i) = poly_coeffs(1);
        end
    end
end