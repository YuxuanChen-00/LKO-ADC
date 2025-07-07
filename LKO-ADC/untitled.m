% =========================================================================
% 1. 设置与加载模型 (Setup and Load Model) [1](@ref)
% =========================================================================
clear; clc;

% --- 指定您的 ONNX 模型文件路径
onnxFilePath = 'model.onnx';

% --- 检查文件是否存在
if ~isfile(onnxFilePath)
    error("ONNX模型文件未找到！请检查路径: '%s'", onnxFilePath);
end

% --- 导入ONNX网络
disp('正在加载ONNX模型...');
try
    net = importNetworkFromONNX(onnxFilePath);
    analyzeNetwork(net)
    disp('模型加载成功！');
    
    % =========================================================================
    % 关键修改：检查模型输入结构 [1,7](@ref)
    % =========================================================================
    disp('网络输入信息:');
    disp(net.InputNames);
    inputCount = numel(net.InputNames);
    disp(['输入数量: ', num2str(inputCount)]);
    
    % 获取每个输入的期望尺寸
    disp('每个输入的期望尺寸:');
    for i = 1:inputCount
        inputSize = net.getInputSize(i);
        fprintf('输入 %d (%s): %s\n', i, net.InputNames{i}, mat2str(inputSize));
    end
    
catch ME
    disp('模型加载失败。错误信息:');
    rethrow(ME);
end

% (可选) 分析网络结构，查看输入层的名称和顺序
% analyzeNetwork(net);


% =========================================================================
% 2. 准备输入数据 (Prepare Input Data) [1,6,7,8](@ref)
% =========================================================================
disp('正在准备输入数据...');

% --- 定义模型输入的维度参数
% 这些参数必须与您导出模型时使用的参数完全一致
state_size = 6;
control_size = 6;
delay_step = 9;
batch_size = 1; % 我们测试单次推理，所以batch为1

% --- 创建四个输入的模拟数据 (使用全零，与Python示例一致)
% 注意: MATLAB深度学习的典型数据格式是 (通道C, 批次B, 时间T)
state_current_data = zeros(state_size, batch_size);         % 尺寸: [6, 1]
control_current_data = zeros(control_size, batch_size);     % 尺寸: [6, 1]
state_sequence_data = zeros(state_size, delay_step - 1, batch_size);    % 尺寸: [6, 8, 1]
control_sequence_data = zeros(control_size, delay_step - 1, batch_size);% 尺寸: [6, 8, 1]

% --- 将常规的MATLAB数组转换为 dlarray 对象
% dlarray 是 Deep Learning Toolbox 的专用数组格式，可以携带维度信息
% 'C' = Channel/Feature, 'B' = Batch, 'T' = Time/Sequence
state_current = dlarray(state_current_data, 'CB');
control_current = dlarray(control_current_data, 'CB');
state_sequence = dlarray(state_sequence_data, 'CBT');
control_sequence = dlarray(control_sequence_data, 'CBT');

% =========================================================================
% 关键修改：处理额外的输入（如隐藏状态）[1,7](@ref)
% =========================================================================
% 创建额外的输入（如果需要）
if inputCount > 4
    % 获取第五个输入的期望尺寸
    hidden_size = net.getInputSize(5);
    
    % 根据期望尺寸创建隐藏状态数据
    % 注意：这里假设隐藏状态是二维数据（特征x批次）
    hidden_state_data = zeros(hidden_size(1), batch_size);
    hidden_state = dlarray(hidden_state_data, 'CB');
    
    fprintf('已创建隐藏状态输入，尺寸: [%d, %d]\n', hidden_size(1), batch_size);
end

disp('输入数据准备完毕。');


% =========================================================================
% 3. 执行推理并获取结果 (Run Inference and Get Results) [1,6,7](@ref)
% =========================================================================
disp('执行模型前向传播...');

% --- 调用 predict 函数进行推理
% **关键**: 必须按照导出时 input_names 的顺序传入所有输入
tic; % 开始计时

% =========================================================================
% 关键修改：按输入名称顺序传递所有输入 [1,7](@ref)
% =========================================================================
switch inputCount
    case 4
        % 只有4个输入的情况
        pred_dlarray = predict(net, state_current, control_current, state_sequence, control_sequence);
        
    case 5
        % 有5个输入的情况（包含隐藏状态）
        pred_dlarray = predict(net, state_current, control_current, state_sequence, control_sequence, hidden_state);
        
    otherwise
        error('不支持的输入数量: %d。请检查模型结构。', inputCount);
end

elapsedTime = toc; % 结束计时

% --- 从 dlarray 中提取常规的数值结果
output_pred = extractdata(pred_dlarray);

fprintf('单次推理完成！耗时: %.6f 秒\n', elapsedTime);

% --- 显示输出结果的尺寸
disp('输出结果 "pred" 的尺寸:');
disp(size(output_pred));

% --- 显示部分输出结果
disp('输出结果 "pred" 的前10个值:');
disp(output_pred(1:10));

% =========================================================================
% 4. 验证输出尺寸 [6,8](@ref)
% =========================================================================
% 获取输出层信息
outputNames = net.OutputNames;
outputCount = numel(outputNames);
disp('网络输出信息:');
disp(outputNames);

% 检查输出尺寸是否符合预期
for i = 1:outputCount
    outputSize = net.getOutputSize(i);
    fprintf('输出 %d (%s) 的实际尺寸: %s\n', i, outputNames{i}, mat2str(size(output_pred)));
    
    % 验证输出尺寸是否符合模型预期
    if ~isequal(size(output_pred), outputSize)
        warning('输出 %d 的尺寸不匹配! 预期: %s, 实际: %s', ...
            i, mat2str(outputSize), mat2str(size(output_pred)));
    end
end