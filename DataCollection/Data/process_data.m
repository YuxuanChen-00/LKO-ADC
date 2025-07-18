%% 离线数据清洗工具
% 作者：MATLAB助手 | 日期：2025-05-27
% 功能：异常值检测 + 噪声滤波 + 数据保存
% 输入：inputPath（原始数据路径），outputPath（结果路径）

% ===== 用户配置区域 =====
inputPath = 'MotionData9\RawData\50secTest';
outputPath = 'MotionData9\FilteredData\50secTest';
targetExtension = '*.mat';

% 滤波参数
filterMethod = 'MovingAverage'; % 可选: MovingAverage/Median
mean_windowSize = 1;                % 滤波窗口大小(奇数)
med_windowSize = 1;

% 绘图参数
plotRow = 1;           % 绘制第几行数据
lineWidth = 1.5;       % 曲线线宽
timeUnit = 's';        % 时间轴单位

% ===== 主程序开始 =====
% 创建输出目录
if ~exist(outputPath, 'dir')
    mkdir(outputPath);
end

% 获取文件列表
fileList = dir(fullfile(inputPath, targetExtension));

% 批量处理循环
for i = 1:length(fileList)
    try
        % 加载数据
        filePath = fullfile(inputPath, fileList(i).name);
        data = load(filePath);
        originalState = data.state;  % 原始数据备份
        
        % 替换Inf异常值
        if isstruct(data) && isfield(data, 'state')
            originalState(isinf(originalState)) = NaN;  % 替换Inf为NaN
            originalState = fillmissing(originalState, 'linear',2); % 线性插值[1](@ref)
        end


        % 中值滤波
        state = my_medfilt1(originalState, med_windowSize);

        % 均值滤波
        state = movmean(state, mean_windowSize, 2);
        


        % 保存处理结果
        data.state = state;
        save(fullfile(outputPath, fileList(i).name), '-struct', 'data');
        
        % % ===== 绘制叠加对比图 =====
        % [n, t] = size(state);
        % timeAxis = (0:t-1);      % 生成时间轴
        % 
        % figure('Name','滤波效果验证','NumberTitle','off')
        % % 原始信号绘制（红色虚线）
        % plot(timeAxis, originalState(plotRow,:), 'r--',...
        %     'LineWidth',lineWidth, 'DisplayName','原始数据')
        % hold on
        % % 滤波信号绘制（蓝色实线）
        % plot(timeAxis, state(plotRow,:), 'b-',...
        %     'LineWidth',lineWidth, 'DisplayName','滤波后数据')
        % hold off
        % 
        % % 图形美化
        % title(sprintf(fileList(i).name, '信号对比 (%s滤波)',  filterMethod))
        % xlabel(['时间 (', timeUnit, ')'])
        % ylabel('幅值')
        % legend('Location','best')
        % grid on
        % set(gca, 'FontSize',12)  % 统一字体大小
        % set(gcf,'Position',[200 200 800 400])  % 设置图像尺寸
        
        fprintf('成功处理: %s\n', fileList(i).name);
        
    catch ME
        warning('文件%s处理失败: %s', fileList(i).name, ME.message);
    end
end