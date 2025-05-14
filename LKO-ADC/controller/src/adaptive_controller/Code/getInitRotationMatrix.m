% 将vicon中的世界坐标系转换到机器人基座标系下
function init_rotation_matrix = getInitRotationMatrix(raw_data)
    world_base = raw_data(1:6);
    world_bend1 = raw_data(7:12);
    world_bend2 = raw_data(13:18);
    world_end = raw_data(19:24);
    [~, rm_base_bend1] = CoordinateTransfer(world_base, world_bend1);
    [~, rm_base_bend2] = CoordinateTransfer(world_base, world_bend2);
    [~, rm_base_end] = CoordinateTransfer(world_base,world_end);
    init_rotation_matrix.R1 = rm_base_bend1;
    init_rotation_matrix.R2 = rm_base_bend2;
    init_rotation_matrix.R3 = rm_base_end;