sorted_results = readMyDataFile('LKO_hyperparameter_search_denorm_eval.txt');

for i = 1:size(sorted_results, 2)
    current_dimenson = sorted_results(i).dimension;
    current_delay = sorted_results(i).delay;
    sorted_results(i).dimension = current_dimenson/current_delay;
end

save('full_results.mat', "sorted_results")

function dataStruct = readMyDataFile(filename)
% READMYDATAFILE 从指定的文本文件中读取实验数据并将其存入结构体数组.
%
%   S = READMYDATAFILE(FILENAME) 读取名为 FILENAME 的文件,
%   将每一行解析为包含 delay, dimension, 和 mean_rmse 字段的结构体,
%   并返回一个结构体数组 S.

    % 打开文件进行读取
    fileID = fopen(filename, 'r');
    if fileID == -1
        error('无法打开文件: %s', filename);
    end

    % 初始化一个空的结构体数组
    dataStruct = struct('delay', {}, 'dimension', {}, 'mean_rmse', {});

    % 跳过标题行
    fgetl(fileID); % 跳过第一行
    fgetl(fileID); % 跳过第二行分割线

    % 逐行读取文件
    while ~feof(fileID)
        line = fgetl(fileID);
        
        % 如果是空行或者分割线行，则跳过
        if isempty(line) || all(line == '-')
            continue;
        end
        
        % 使用sscanf解析数据
        % 格式: 排名 | delay | PhiDimensions | Loss (RMSE) | 文件夹
        % 我们只需要 delay, PhiDimensions, 和 Loss
        C = sscanf(line, '%*d | %d | %d | %f | %*s');
        
        % 检查是否成功解析出3个数值
        if numel(C) == 3
            % 创建一个临时结构体
            tempStruct.delay = C(1);
            tempStruct.dimension = C(2);
            tempStruct.mean_rmse = C(3);
            
            % 将临时结构体追加到主结构体数组中
            dataStruct(end+1) = tempStruct;
        end
    end

    % 关闭文件
    fclose(fileID);

end