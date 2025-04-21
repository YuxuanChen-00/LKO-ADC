function [current_state, is_valid, last_sample] = sampleAndFilterViconData(samplingRate, num_samples, num_state, initRotationMatrix, weight, last_sample)
    global onemotion_data;
    is_valid = 1;
    p = 1;
    valid_samples = 0; % 有效采样计数器
    sample_buffer = zeros(num_state, num_samples);
    raw_data = []; % 用于存储原始数据
    
    % 动态阈值参数配置
    DISTANCE_THRESHOLD = 0.15; % 欧氏距离阈值（单位：米）
    ANGLE_THRESHOLD = 15;      % 角度变化阈值（单位：度）
    MAX_CONSECUTIVE_FAILS = 3;% 最大连续异常次数
    
    consecutive_fails = 0;     % 连续异常计数器
    
    while valid_samples < num_samples
        waitfor(samplingRate);
        
        % 获取新采样点
        new_sample = transferVicon2Base(onemotion_data, initRotationMatrix);
        raw_data(:,p) = onemotion_data;
        p = p + 1;

        % 异常检测流程
        [is_outlier, reason] = checkOutlier(new_sample, last_sample,...
                                         DISTANCE_THRESHOLD, ANGLE_THRESHOLD);
        
        if ~is_outlier
            % 存入有效样本
            sample_buffer(:, mod(valid_samples, num_samples)+1) = new_sample;
            valid_samples = valid_samples + 1;
            last_sample = new_sample; % 更新有效样本
            consecutive_fails = 0;
        else
            % 异常处理
            consecutive_fails = consecutive_fails + 1;
            fprintf('异常点过滤: %s (连续异常次数: %d)\n', reason, consecutive_fails);
            
            if consecutive_fails >= MAX_CONSECUTIVE_FAILS
                is_valid = 0;
                error('连续%d次采样异常，系统终止采样', MAX_CONSECUTIVE_FAILS);
            end
        end
    end

    % 计算加权平均值（仅使用有效样本）
    weight = weight(1:valid_samples) / sum(weight(1:valid_samples));
    current_state = sample_buffer(:, 1:valid_samples) * weight(:);
    
    % 最终有效性验证
    if any(isnan(current_state)) || valid_samples < num_samples*0.5
        warning('有效样本不足: %d/%d', valid_samples, num_samples);
        is_valid = 0;
    end
end