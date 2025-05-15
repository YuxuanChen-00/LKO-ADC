input_folder = 'MotionData';
output_folder = 'RelativeMotionData';
mkdir(output_folder); % 自动创建输出目录
files = dir(fullfile(input_folder, '*.mat')); % 获取所有.mat文件列表

list = [1,2,3,7,8,9];
for i = 1:length(files)
    % 加载文件并保留所有原始变量[8](@ref)
    file_path = fullfile(input_folder, files(i).name);
    data = load(file_path); % 数据以结构体形式加载
    
    % 提取原始state矩阵的第一列
    original_first_col = data.state(:, 1);
    
    % 修改第一列的指定通道为0[9](@ref)
    data.state([1,2,3,7,8,9], 1) = 0;
    
    % 处理后续列（减去原始第一列对应值）
    for col = 2:size(data.state, 2)
        data.state(list, col) = data.state(list, col) - original_first_col(list);
    end
    
    % 保存到新路径（保留其他变量）[6](@ref)
    save(fullfile(output_folder, files(i).name), '-struct', 'data');
end