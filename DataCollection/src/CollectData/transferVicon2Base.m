function x_meas = transferVicon2Base(raw_data, init_rotation_matrix, init_position)
    % 检查数据长度是否为6的倍数
    if mod(length(raw_data), 6) ~= 0
        error('输入数据长度必须是6的倍数');
    end
    
    % 计算总刚体数量（包含基准坐标系）
    total_bodies = length(raw_data)/6;
    if total_bodies < 2
        error('至少需要2个刚体（基准+至少一个刚体）');
    end
    
    % 提取基准坐标系
    world_base = raw_data(1:6);
    
    % 获取需要处理的刚体数量（扣除基准）
    num_bodies = total_bodies - 1;
    
    % 验证旋转矩阵字段数量匹配
    rotation_fields = fieldnames(init_rotation_matrix);
    if numel(rotation_fields) ~= num_bodies
        error('旋转矩阵字段数量与刚体数量不匹配');
    end
    
    % 预分配存储空间
    x_meas = zeros(num_bodies*6, 1); % 每个刚体贡献6个数据
    
    % 循环处理每个刚体
    for i = 1:num_bodies
        % 计算数据索引范围
        start_idx = 6*i + 1;
        end_idx = 6*(i+1);
        world_body = raw_data(start_idx:end_idx);
        
        field_name_rot = ['R' num2str(i)];
        field_name_pos = ['P' num2str(i)];
        
        % 坐标系转换
        [post, rm] = CoordinateTransfer(world_base, world_body);
        post = post - init_position.(field_name_pos);
        % 旋转矩阵补偿（使用动态字段名）

        % rm_compensated = inv(init_rotation_matrix.(field_name_rot)) * rm;
        rm_compensated = rm;
        
        % 提取对角线方向分量
        direction = [rm_compensated(1,1); 
                    rm_compensated(2,2);
                    rm_compensated(3,3)];
        
        % 填充到结果数组
        idx_range = (i-1)*6 + 1 : i*6;
        x_meas(idx_range) = [post; direction];
    end
end