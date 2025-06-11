function [file_control, file_state, file_label] = generate_sequence_data(control, states, time_step)
    if nargin < 4
        pred_step = 1; % 
    end

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
    file_state = zeros(d, num_samples, time_step);
    file_label = zeros(d, num_samples, time_step);
    
    % 构建时间窗口
    for sample_idx = 1:num_samples
        time_window = sample_idx + time_step - 1:-1:sample_idx;
        
        % 当前状态序列 [s(t) ... s(t+m-1)]
        file_state(:, sample_idx, :) = states(:, time_window);
 
        % 标签序列 [s(t+1) ... s(t+m)]
        file_label(:, sample_idx, :) = states(:, time_window + 1);

        % 控制序列
        file_control(:, sample_idx) = control(:, sample_idx + time_step - 1);
    end

end