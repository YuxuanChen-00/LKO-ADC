raw_data = load('raw_data.mat');
raw_data = raw_data.raw_data;
raw_index = 1;
num_state = 18;
length = 1200;  % 数据长度

num_samples = 4;  % 每4个采样点做平均
sample_buffer = zeros(num_state, num_samples);  % 用于存储采样值的缓冲区
sample_index = 1;  % 缓冲区索引

% 获得初始旋转矩阵
initRotationMatrix = getInitRotationMatrix(raw_data(:,1)');

% 初始化数组记录当前时刻位置和控制输入
state = zeros(num_state, length);
raw_state = zeros(num_state, length);
median_window_size = 60;
median_window = zeros(num_state, median_window_size);
for k = 1:length
    % 进行4次采样，更新采样缓冲区
    for i = 1:num_samples
        % 获取一个新的采样点=
        new_sample = Raw2Xmeas(raw_data(:,raw_index)', initRotationMatrix);
        raw_index = raw_index + 1;
        % 将新采样值存入缓冲区
        sample_buffer(:, sample_index) = new_sample;
        sample_index = sample_index + 1;  % 更新索引

        % 如果缓冲区已满，重新从头开始
        if sample_index > num_samples
            sample_index = 1;
        end
    end
    
    % 采样区间异常点检测
    weight = [0.1, 0.2, 0.3, 0.4]';
    weight = weight/sum(weight);
    
    % 计算当前状态（4个采样点的加权平均值）
    current_state = sample_buffer*weight;  % 计算当前状态（4个采样点的平均值）
    raw_state(:,k) = current_state;
    % 当前状态异常点检测
    threshold = [5,5,0.5,0.01,0.01,0.01,...
        5,5,2,0.03,0.03,0.03,...
        5,5,2,0.1,0.1,0.1];
    
    if k > 1
        for i = 1:num_state
            if abs(current_state(i) - state(i,k-1)) > threshold(i) 
                current_state(i) = state(i,k-1);
            end
        end
    end
    
    state(:, k) = current_state;
end
figure(1)
for p = 1:6
    subplot(6,1,p);
    plot(state(p,:));
end

figure(2)
for p = 7:12
    subplot(6,1,p-6);
    plot(state(p,:));
end

figure(3)
for p = 13:18
    subplot(6,1,p-12);
    plot(state(p,:));
end

figure(4)
for p = 1:6
    subplot(6,1,p);
    plot(raw_state(p,:));
end

figure(5)
for p = 7:12
    subplot(6,1,p-6);
    plot(raw_state(p,:));
end

figure(6)
for p = 13:18
    subplot(6,1,p-12);
    plot(raw_state(p,:));
end


