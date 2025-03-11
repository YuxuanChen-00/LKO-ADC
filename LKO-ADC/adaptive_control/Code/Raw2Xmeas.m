% 将vicon中的世界坐标系转换到机器人基座标系下
function x_meas = Raw2Xmeas(raw_data, init_rotation_matrix)
    world_base = raw_data(1:6);
    world_bend1 = raw_data(7:12);
    world_bend2 = raw_data(13:18);
    world_end = raw_data(19:24);
    [post_base_bend1, rm_base_bend1] = CoordinateTransfer(world_base, world_bend1);
    [post_base_bend2, rm_base_bend2] = CoordinateTransfer(world_base, world_bend2);
    [post_base_end, rm_base_end] = CoordinateTransfer(world_base,world_end);
    
    % 将该刚体旋转矩阵调整为初始时刻刚体和基座标对齐后的旋转矩阵
    rm_base_bend1 = inv(init_rotation_matrix.R1)*rm_base_bend1;
    rm_base_bend2 = inv(init_rotation_matrix.R2)*rm_base_bend2;
    rm_base_end = inv(init_rotation_matrix.R3)*rm_base_end;
    
    direction_base_bend1 = [rm_base_bend1(1,1); rm_base_bend1(2,2); rm_base_bend1(3,3)];
    direction_base_bend2 = [rm_base_bend2(1,1); rm_base_bend2(2,2); rm_base_bend2(3,3)];
    direction_base_end = [rm_base_end(1,1); rm_base_end(2,2); rm_base_end(3,3)];
    base_bend1 = [post_base_bend1; direction_base_bend1];
    base_bend2 = [post_base_bend2; direction_base_bend2];
    base_end = [post_base_end; direction_base_end];
    x_meas = [base_bend1; base_bend2; base_end];
end