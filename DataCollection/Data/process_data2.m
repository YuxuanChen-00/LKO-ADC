outputir = 'MotionData5\RawData\50secTest';
inputdata_path = 'MotionData4\RawData\50secTest';
file_list= dir(fullfile(inputdata_path, '*.mat'));
num_files = length(file_list);
num_tracker = 3;
num_state = (num_tracker-1)*6;
lastAoData = [0,0,0,0,0,0,0];
weight = [1,2,3,4,5]';
num_samples = 5;  % 每3个采样点做平均

for file_idx = 1:num_files
    file_name = file_list(file_idx).name;
    file_path = fullfile(inputdata_path, file_name);
    current_data = load(file_path);
    raw_data = current_data.raw_data;
    input = current_data.input;

    signal_length = size(current_data.input, 2);
    state = zeros(num_state, signal_length);

    onemotion_data = raw_data(:, 1);
    last_sample = transferVicon2Base2(onemotion_data);

    for k = 1:signal_length
       current_raw_data = raw_data(:, (k-1)*num_samples+1:k*num_samples);
       [current_state, last_sample] = sampleAndFilterViconData2(current_raw_data, num_samples, weight, last_sample);
       state(:, k) = current_state;
    end

    output_file_path = fullfile(outputir, file_name);
    save(output_file_path, "raw_data", "input", "state");
end


function [current_state, last_sample] = sampleAndFilterViconData2(raw_data, num_samples, weight, last_sample)
    valid_samples = []; % 有效采样计数器
    num_bodies = 2;
    sample_buffer = zeros(num_bodies*6, num_samples);
    
    % 动态阈值参数配置
    DISTANCE_THRESHOLD = 40; % 欧氏距离阈值（单位：米）
    ANGLE_THRESHOLD = 8;      % 角度变化阈值（单位：度）
    
    consecutive_fails = 0;     % 连续异常计数器
    
    for i= 1 : num_samples
        onemotion_data = raw_data(:, i);

        % 获取新采样点
        new_sample = transferVicon2Base2(onemotion_data);

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
            sample_buffer(:,i) = Inf;
            fprintf('异常点过滤: %s (连续异常次数: %d)\n', reason, consecutive_fails);
            continue ;
        end

    end

    if isempty(valid_samples)
        current_state = inf*ones(size(new_sample));
    else
        weight = weight(valid_samples) / sum(weight(valid_samples));
        current_state = sample_buffer(:, valid_samples) * weight;
    end
end


function x_meas = transferVicon2Base2(raw_data)
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
        % 旋转矩阵
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

function [Post_base_target, Eular_base_target] = CoordinateTransfer(world_base, world_target)
    Rworld2base = Euler2RotationFunc(world_base(4:6));
    Rworld2target = Euler2RotationFunc(world_target(4:6));

    Post_base_target= Rworld2base*(world_target(1:3)-world_base(1:3));
    Eular_base_target = inv(Rworld2target)*Rworld2base;
    function R = Euler2RotationFunc(euler)
        %XYZ_Euler to Rotation matrix
        x=euler(1);y=euler(2);z=euler(3);
        Rz=[cos(z),-sin(z),0;...
            sin(z),cos(z), 0;...
            0,     0     , 1];
        Ry=[cos(y), 0, sin(y);...
            0,      1,     0;...
            -sin(y),0, cos(y)];
        Rx=[1,     0,       0;...
            0, cos(x), -sin(x);....
            0, sin(x), cos(x)];
        R=(Rx*Ry*Rz)';
    end
end

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
