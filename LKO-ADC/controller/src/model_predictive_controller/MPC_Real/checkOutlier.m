% 异常检测子函数
function [is_outlier, reason] = checkOutlier(new_sample, last_sample, dist_th, ang_th)
    is_outlier = false;
    reason = '';
    
    num_bodies = size(new_sample)/6;
    for i = 0:num_bodies-1
        start_index = i*6 +  1;
        end_index = (i + 1)*6;
        current_new_sample = new_sample(start_index:end_index);
        current_last_sample = last_sample(start_index:end_index);
        % 位置突变检测（前3个元素为位置）
        position_delta = norm(current_new_sample(1:3) - current_last_sample(1:3));
        if position_delta > dist_th
            is_outlier = true;
            reason = sprintf('第%d个刚体位置突变(%.2fmm > %.2fmm)', i+1, position_delta, dist_th);
            return;
        end
        
        % 方向突变检测（后3个元素为方向向量）
        old_dir = current_last_sample(4:6);
        new_dir = current_new_sample(4:6);
        angle_change = rad2deg(acos(dot(old_dir,new_dir)/(norm(old_dir)*norm(new_dir))));
        
        if angle_change > ang_th
            is_outlier = true;
            reason = sprintf('第%d个刚体方向突变(%.1f° > %.1f°)', i+1, angle_change, ang_th);
            return;
        end
    end
    
end