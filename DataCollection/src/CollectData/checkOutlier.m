% 异常检测子函数
function [is_outlier, reason] = checkOutlier(new_sample, last_sample, dist_th, ang_th)
    is_outlier = false;
    reason = '';
    
    % 位置突变检测（前3个元素为位置）
    position_delta = norm(new_sample(1:3) - last_sample(1:3));
    if position_delta > dist_th
        is_outlier = true;
        reason = sprintf('位置突变(%.2fm > %.2fm)', position_delta, dist_th);
        return;
    end
    
    % 方向突变检测（后3个元素为方向向量）
    old_dir = last_sample(4:6);
    new_dir = new_sample(4:6);
    angle_change = rad2deg(acos(dot(old_dir,new_dir)/(norm(old_dir)*norm(new_dir))));
    
    if angle_change > ang_th
        is_outlier = true;
        reason = sprintf('方向突变(%.1f° > %.1f°)', angle_change, ang_th);
        return;
    end
    
    % 物理合理性检查（示例：速度突变）
    if ~isempty(last_sample)
        dt = 1/80; % 80Hz采样周期
        velocity = (new_sample(1:3) - last_sample(1:3)) / dt;
        if any(abs(velocity) > 5) % 速度超过5m/s视为异常
            is_outlier = true;
            reason = sprintf('速度异常(%.1f m/s)', max(abs(velocity)));
        end
    end
end