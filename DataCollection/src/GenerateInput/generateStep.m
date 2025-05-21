function allSignals = generateStep(numChannels, max_vals_per_channel, min_vals_per_channel, interpSteps, holdSteps)
% generateMultiChannelStepSignals - 生成用于系统辨识的多通道阶跃激励信号
%
%   allSignals = generateMultiChannelStepSignals(numChannels, max_vals_per_channel, min_vals_per_channel, interpSteps, holdSteps)
%   为多通道系统生成一系列激励信号。每个通道可以有其独立的最小和最大信号值。
%   信号在单个通道和多个通道联合激励之间切换，遍历所有可能的通道激活组合（不包括所有通道都关闭的情况）。
%   当信号从其通道的低电平切换到高电平或反之时，使用线性插值，并在达到目标电平后保持一段时间。
%
% 输入参数:
%   numChannels (integer):          系统输入的通道数量 (例如: 6)
%   max_vals_per_channel (1xN double): 每个通道的最大输入值向量 (例如: [5,5,5,5,5,5])
%                                      N 必须等于 numChannels。
%   min_vals_per_channel (1xN double): 每个通道的最小输入值向量 (例如: [0,0,0,0,0,0])
%                                      N 必须等于 numChannels。
%   interpSteps (integer):          从一个值转换到另一个值时的线性插值步数 (例如: 10)
%                                    注意：插值本身包含 interpSteps 个点，即 ramp 的长度。
%   holdSteps (integer):            在插值达到目标值后，保持该值的步数 (例如: 10)
%
% 输出参数:
%   allSignals (MxN double):        生成的信号矩阵，其中 M 是总时间步数，N 是通道数。
%                                    每一列代表一个通道的信号，每一行代表一个时间步。
%
% 示例:
%   nCh = 3;
%   max_p = [5, 6, 5.5]; % 通道特定最大值
%   min_p = [0, 1, 0.5]; % 通道特定最小值
%   iSteps = 10;
%   hSteps = 10;
%   signals = generateMultiChannelStepSignals(nCh, max_p, min_p, iSteps, hSteps);
%   figure;
%   plot(signals);
%   xlabel('时间步长');
%   ylabel('信号值');
%   title(sprintf('系统辨识激励信号 (%d通道, %d步插值, %d步保持)', ...
%                 nCh, iSteps, hSteps));
%   legend_entries = arrayfun(@(x) sprintf('通道 %d (%.1f-%.1f)', x, min_p(x), max_p(x)), 1:nCh, 'UniformOutput', false);
%   legend(legend_entries);

% --- 输入验证 ---
if ~isscalar(numChannels) || numChannels <= 0 || floor(numChannels) ~= numChannels
    error('numChannels 必须是一个正整数标量。');
end

if ~isnumeric(max_vals_per_channel) || ~isvector(max_vals_per_channel) || length(max_vals_per_channel) ~= numChannels
    error('max_vals_per_channel 必须是一个长度为 numChannels 的数值向量。');
end
if ~isnumeric(min_vals_per_channel) || ~isvector(min_vals_per_channel) || length(min_vals_per_channel) ~= numChannels
    error('min_vals_per_channel 必须是一个长度为 numChannels 的数值向量。');
end

if any(min_vals_per_channel >= max_vals_per_channel)
    error('对于每个通道，其最小值 min_vals_per_channel(i) 必须严格小于其最大值 max_vals_per_channel(i)。');
end

if ~isscalar(interpSteps) || interpSteps <= 0 || floor(interpSteps) ~= interpSteps
    error('interpSteps 必须是一个正整数标量。');
end

if ~isscalar(holdSteps) || holdSteps < 0 || floor(holdSteps) ~= holdSteps
    error('holdSteps 必须是一个非负整数标量。');
end

% --- 计算单个脉冲的总步长 ---
% 这个步长对于所有通道的脉冲都是相同的，因为 interpSteps 和 holdSteps 是全局的
% 1. 插值上升 (interpSteps)
% 2. 保持高位 (holdSteps)
% 3. 插值下降 (interpSteps)
% 4. 保持低位 (holdSteps)
pulse_duration = interpSteps + holdSteps + interpSteps + holdSteps;

% --- 计算总共需要多少种激励组合 ---
% 我们要遍历激活1个通道, 2个通道, ..., numChannels个通道的所有组合
numCombinations = 0;
for k = 1:numChannels
    numCombinations = numCombinations + nchoosek(numChannels, k);
end

% 如果没有组合（例如 numChannels = 0，尽管已被验证排除），或者脉冲持续时间为0，则返回空
if numCombinations == 0 || pulse_duration == 0
    allSignals = zeros(numChannels, 0); % 返回正确维度的空矩阵
    return;
end

% --- 预分配内存以提高效率 ---
totalSignalSteps = numCombinations * pulse_duration;
allSignals = zeros(numChannels, totalSignalSteps);

% --- 生成信号 ---
current_row_start = 1; % 用于填充 allSignals 的起始行索引

% 外层循环: 激活通道的数量 (从1到numChannels)
for k = 1:numChannels
    % 获取当前数量下所有可能的通道组合
    % combs 是一个矩阵，每一行是一个组合，列出了要激活的通道索引
    combs = nchoosek(1:numChannels, k);
    
    numCurrentCombs = size(combs, 1);
    
    % 内层循环: 遍历当前数量下的每一种特定组合
    for i = 1:numCurrentCombs
        current_combination_indices = combs(i, :); % 当前要激活的通道索引
        
        % 为这个组合创建一个信号块
        current_block = zeros(pulse_duration, numChannels);
        
        for ch = 1:numChannels
            % 获取当前通道的低电平和高电平值
            ch_lowVal = min_vals_per_channel(ch);
            ch_highVal = max_vals_per_channel(ch);
            
            % 定义当前通道的“激活脉冲”和“非激活”波形
            interp_up_segment = linspace(ch_lowVal, ch_highVal, interpSteps)';
            hold_high_segment = repmat(ch_highVal, holdSteps, 1);
            interp_down_segment = linspace(ch_highVal, ch_lowVal, interpSteps)';
            hold_low_segment = repmat(ch_lowVal, holdSteps, 1);
            
            active_channel_pulse_for_ch = [interp_up_segment; hold_high_segment; interp_down_segment; hold_low_segment];
            inactive_channel_waveform_for_ch = repmat(ch_lowVal, pulse_duration, 1);
            
            if ismember(ch, current_combination_indices)
                % 如果此通道在当前组合中被激活
                current_block(:, ch) = active_channel_pulse_for_ch;
            else
                % 如果此通道不在此组合中 (保持其低电平)
                current_block(:, ch) = inactive_channel_waveform_for_ch;
            end
        end
        
        % 将此信号块添加到总信号矩阵中
        current_row_end = current_row_start + pulse_duration - 1;
        allSignals(:, current_row_start:current_row_end) = current_block';
        
        % 更新下一个块的起始行
        current_row_start = current_row_end + 1;
    end
end

% --- 函数结束 ---
end