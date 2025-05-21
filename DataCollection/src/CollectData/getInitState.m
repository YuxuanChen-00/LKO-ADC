function [init_rotation_matrix, init_position] = getInitState(raw_data)
    % ������ݳ����Ƿ�Ϊ6�ı���
    if mod(length(raw_data), 6) ~= 0
        error('�������ݳ��ȱ�����6�ı���');
    end
    
    % ��������������۳���һ����׼����ϵ��
    num_bodies = length(raw_data)/6 - 1;
    if num_bodies < 1
        error('������Ҫ�����������ݣ���׼+����һ�����壩');
    end
    
    % ��ȡ��׼����ϵ����
    world_base = raw_data(1:6);
    
    % ��̬����ÿ������
    for i = 1:num_bodies
        % ���㵱ǰ�������������
        start_idx = 6*i + 1;
        end_idx = 6*(i+1);
        world_body = raw_data(start_idx:end_idx);
        
        % ����ת�����洢���
        [position, rm] = CoordinateTransfer(world_base, world_body);
        init_rotation_matrix.(['R' num2str(i)]) = rm;
        init_position.(['P' num2str(i)]) = position;
    end
end