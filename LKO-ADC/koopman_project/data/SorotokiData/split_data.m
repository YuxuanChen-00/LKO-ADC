%% 设置路径
sourceDir = 'testData\';      % 替换为你的.mat文件所在目录
targetDir = 'testData\testdata';    % 替换为目标保存目录

%% 创建目标文件夹（如果不存在）
if ~exist(targetDir, 'dir')
    mkdir(targetDir);
end

%% 获取所有.mat文件列表
fileList = dir(fullfile(sourceDir, '*.mat'));

%% 循环处理每个文件
for iFile = 1:length(fileList)
    % 加载当前文件
    fileName = fileList(iFile).name;
    filePath = fullfile(sourceDir, fileName);
    fileData = load(filePath);  % 加载所有变量
    
    % 获取变量名列表
    vars = fieldnames(fileData);
    
    % 遍历每个变量进行处理
    for iVar = 1:length(vars)
        currentVar = fileData.(vars{iVar});
        
        % 仅处理二维矩阵（跳过结构体/单元格等）
        if ismatrix(currentVar) && isnumeric(currentVar)
            % 去除全零行
            nonZeroRows = any(currentVar ~= 0, 2);
            fileData.(vars{iVar}) = currentVar(nonZeroRows, :);
        end
    end
    
    % 保存到新路径（保持原文件名）
    savePath = fullfile(targetDir, fileName);
    save(savePath, '-struct', 'fileData');
    fprintf('已处理: %s\n', fileName);
end

disp('所有文件处理完成！');