function [current_state, raw_data, last_sample] = sampleAndFilterViconData(samplingRate, num_samples, initRotationMatrix, initPosition, weight, last_sample)
    global onemotion_data;
    p = 1;
    valid_samples = []; % 有效采样计数器
    num_bodies = numel(fieldnames(initRotationMatrix));
    sample_buffer = zeros(num_bodies*6, num_samples);
    raw_data = []; % 用于存储原始数据
    
    % 动态阈值参数配置
    DISTANCE_THRESHOLD = 100; % 欧氏距离阈值（单位：米）
    ANGLE_THRESHOLD = 20;      % 角度变化阈值（单位：度）
    
    consecutive_fails = 0;     % 连续异常计数器
    
    for i = 1:num_samples
        
        % 获取新采样点
        new_sample = transferVicon2Base(onemotion_data, initRotationMatrix, initPosition);
        raw_data(:,p) = onemotion_data;
        p = p + 1;

        % 异常检测流程
        [is_outlier, reason] = checkOutlier(new_sample, last_sample,...
                                         DISTANCE_THRESHOLD, ANGLE_THRESHOLD);
        
        if ~is_outlier
            % 存入有效样本
            sample_buffer(:, i) = new_sample;
            valid_samples = [valid_samples, i];
            last_sample = new_sample; % 更新有效样本
            consecutive_fails = 0;
        else
            % 异常处理
            consecutive_fails = consecutive_fails + 1;
            sample_buffer(:, i) = Inf;
            fprintf('异常点过滤: %s (连续异常次数: %d)\n', reason, consecutive_fails);
        end

        waitfor(samplingRate);
    end

    % 计算加权平均值（仅使用有效样本）
    if isempty(valid_samples)
        current_state = inf*ones(size(new_sample));
    else
        weight = weight(valid_samples) / sum(weight(valid_samples));
        current_state = sample_buffer(:, valid_samples) * weight;
    end
end