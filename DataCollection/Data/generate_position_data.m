%% 设置路径
sourceDir = 'MotionData3\FilteredData\80minTrain';      % 替换为你的.mat文件所在目录
targetDir = 'MotionData3\FilteredDataPos\80minTrain';    % 替换为目标保存目录

%% 创建目标文件夹（如果不存在）
if ~exist(targetDir, 'dir')
    mkdir(targetDir);
end

%% 获取所有.mat文件列表
fileList = dir(fullfile(sourceDir, '*.mat'));
row = [1,2,3,7,8,9];
%% 循环处理每个文件
for iFile = 1:length(fileList)
    % 加载当前文件
    fileName = fileList(iFile).name;
    filePath = fullfile(sourceDir, fileName);
    fileData = load(filePath);  % 加载所有变量
    
    state = fileData.state;
    fileData.state = state(row, :);
    
    % 保存到新路径（保持原文件名）
    savePath = fullfile(targetDir, fileName);
    save(savePath, '-struct', 'fileData');
    fprintf('已处理: %s\n', fileName);
end

disp('所有文件处理完成！');