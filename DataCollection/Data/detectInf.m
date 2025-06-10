% 选择目标文件夹
folder = uigetdir('MotionData5\RawData\50secTest\');
if folder == 0
    error('未选择文件夹。');
end

% 获取所有.mat文件
files = dir(fullfile(folder, '*.mat'));
if isempty(files)
    error('文件夹中没有.mat文件。');
end

% 初始化结果存储
results = cell(numel(files), 2); % 存储文件名和最大连续Inf步数

% 处理每个文件
for k = 1:numel(files)
    % 加载数据
    filePath = fullfile(folder, files(k).name);
    dataStruct = load(filePath);
    dataField = fieldnames(dataStruct);
    
    % 确保只有一个变量
    % if numel(dataField) > 1
    %     results{k,1} = files(k).name;
    %     results{k,2} = '文件包含多个变量';
    %     continue;
    % end
    
    % 获取数据
    data = dataStruct.state;
    [rows, cols] = size(data);
    
    % 验证数据维度 (12 x t)
    if rows ~= 12
        results{k,1} = files(k).name;
        results{k,2} = '数据行数不为12';
        continue;
    end
    
    % 计算每行的最大连续Inf
    maxContinuousInf = zeros(1, rows);
    
    for r = 1:rows
        % 高效计算连续Inf
        sequence = [0, isinf(data(r,:)), 0];   % 填充首尾零
        diffSeq = diff(sequence);              % 计算差分
        startIdx = find(diffSeq == 1);         % 连续Inf起点
        endIdx = find(diffSeq == -1);           % 连续Inf终点
        lengths = endIdx - startIdx;            % 计算长度
        
        % 记录本行最大连续Inf
        if ~isempty(lengths)
            maxContinuousInf(r) = max(lengths);
        end
    end
    
    % 记录整个文件的最大值
    results{k,1} = files(k).name;
    results{k,2} = max(maxContinuousInf);
end

% 显示结果
disp('检测结果:');
disp('-------------------------------------');
fprintf('%-30s %-20s\n', '文件名', '最大连续Inf步数');
disp('-------------------------------------');
for k = 1:numel(files)
    if isnumeric(results{k,2})
        fprintf('%-30s %-d\n', results{k,1}, results{k,2});
    else
        fprintf('%-30s %-s\n', results{k,1}, results{k,2});
    end
end