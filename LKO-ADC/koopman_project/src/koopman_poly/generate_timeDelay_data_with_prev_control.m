function [file_control, file_state, file_labels] = generate_timeDelay_data_with_prev_control(control, states, time_step)
    % 提取数据并验证维度
    [c, t] = size(control);
    [d, t_check] = size(states);
   

    % 数据一致性检查
    if t ~= t_check
        fprintf('时间步不匹配');
        return;
    end
    if t < time_step+1
        fprintf('时间步不足');
        return;
    end

    % 计算本文件样本数
    num_samples = t - time_step;  % 保证标签窗口有足够数据
    
    % 预分配本文件数据
    file_control = zeros(c,  num_samples);
    file_state = zeros(d*time_step+c*time_step, num_samples);
    file_labels = zeros(d*time_step+c*time_step, num_samples);
    

    % 构建时间窗口
    for sample_idx = 1:num_samples
        time_window = sample_idx : sample_idx + time_step - 1;
        
        % 控制输入序列 [p(t) ... p(t+m-1)]
        file_control(:, sample_idx) = control(:, sample_idx + time_step - 1);
        
        % 当前状态序列 [s(t) ... s(t+m-1)]
        file_state(1:d*time_step, sample_idx) = reshape(states(:, time_window),[],1);

        if time_window(1) == 1
            file_state(d*time_step+1:d*time_step+c, sample_idx) = zeros(size(control, 1),1);
            file_state(d*time_step+c+1:end, sample_idx) = reshape(control(:,time_window(2:end)), [], 1);
        else
            % disp(size(control))
            % disp(time_window - 1)
            file_state(d*time_step+1:end, sample_idx) = reshape(control(:,time_window-1), [], 1);
        end

        % 标签序列 [s(t+1) ... s(t+m)]
        file_labels(1:d*time_step,  sample_idx) = reshape(states(:, time_window + 1),[],1);
        file_labels(d*time_step+1:end,  sample_idx) = reshape(control(:, time_window),[],1);
    end
        
end