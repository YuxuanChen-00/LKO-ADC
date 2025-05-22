function [init_rotation_matrix, init_position] = getInitState(raw_data)
    % 检查数据长度是否为6的倍数
    if mod(length(raw_data), 6) ~= 0
        error('输入数据长度必须是6的倍数');
    end
    
    % 计算刚体总数（扣除第一个基准坐标系）
    num_bodies = length(raw_data)/6 - 1;
    if num_bodies < 1
        error('至少需要两个刚体数据（基准+至少一个刚体）');
    end
    
    % 提取基准坐标系数据
    world_base = raw_data(1:6);
    
    % 动态处理每个刚体
    for i = 1:num_bodies
        % 计算当前刚体的数据区间
        start_idx = 6*i + 1;
        end_idx = 6*(i+1);
        world_body = raw_data(start_idx:end_idx);
        
        % 坐标转换并存储结果
        [position, rm] = CoordinateTransfer(world_base, world_body);
        init_rotation_matrix.(['R' num2str(i)]) = rm;
        init_position.(['P' num2str(i)]) = position;
    end
end