% 创建升维层
hidden_size = 128;
output_size = 64;
customLayer = nn_layer(hidden_size, output_size);

% 创建输入数据（dlarray格式）
inputSize = 50;
batchSize = 32;
X = dlarray(randn(inputSize, batchSize), 'CB'); % 通道×批量

% 前向传播
output = predict(customLayer, X);
size(extractdata(output)) % 应该显示 [64, 32]