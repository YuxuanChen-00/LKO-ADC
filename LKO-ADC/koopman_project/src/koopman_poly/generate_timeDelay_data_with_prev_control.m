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
        state_time_window = sample_idx + time_step - 1 : -1 : sample_idx;
        control_time_window = state_time_window - 1;
        label_time_window = state_time_window + 1;

        % 控制输入序列 [p(t) ... p(t+m-1)]
        file_control(:, sample_idx) = control(:, sample_idx + time_step - 1);

        for i = 1:time_step 
            file_state((i-1)*(d+c)+1:(i-1)*(d+c)+d, sample_idx) = states(:, state_time_window(i));
            file_labels((i-1)*(d+c)+1:(i-1)*(d+c)+d, sample_idx) = states(:, label_time_window(i));
            file_labels((i-1)*(d+c)+1+d:(i-1)*(d+c)+d+c, sample_idx) = control(:, state_time_window(i));
            if control_time_window(i) == 0
                file_state((i-1)*(d+c)+1+d:(i-1)*(d+c)+d+c, sample_idx)= zeros(size(control,1), 1);
            else
                file_state((i-1)*(d+c)+1+d:(i-1)*(d+c)+d+c, sample_idx)= control(:, control_time_window(i));
            end
        end
        % 当前状态序列 [s(t) ... s(t+m-1)]
        % file_state(1:d*time_step, sample_idx) = reshape(states(:, state_time_window),[],1);
        % 
        % if state_time_window(end) == 1
        %     file_state(d*time_step+1:d*time_step+c, sample_idx) = zeros(size(control, 1),1);
        %     file_state(d*time_step+c+1:end, sample_idx) = reshape(control(:,state_time_window(2:end)), [], 1);
        % 
        % else
        %     file_state(d*time_step+1:end, sample_idx) = reshape(control(:,state_time_window-1), [], 1);
        % end

        % 标签序列 [s(t+1) ... s(t+m)]
        % file_labels(1:d*time_step,  sample_idx) = reshape(states(:, state_time_window + 1),[],1);
        % file_labels(d*time_step+1:end,  sample_idx) = reshape(control(:, state_time_window),[],1);
    end
        
end