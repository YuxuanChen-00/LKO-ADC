time_step = 3;
feature_size = 6;         % 特征维度
node_size = 6;            % 节点个数
adjMatrix = [0,1,0,1,0,0;1,0,1,0,1,0;0,1,0,0,0,1;
    1,0,0,0,1,0;0,1,0,1,0,1;0,0,1,0,1,0];           % 邻接矩阵
adjMatrix = adjMatrix + eye(size(adjMatrix, 1));    % 添加自环
D = diag([sum(adjMatrix, 2)]);                      % 度矩阵
adjMatrix = sqrt(inv(D))*adjMatrix*sqrt(inv(D));    % 对称归一化处理

control_size = 6;        % 控制输入维度（根据您的数据调整）
hidden_size = 32;        % LSTM隐藏层维度
PhiDimensions = 68;      % 拼接后的高维特征维度
output_size = PhiDimensions-feature_size*node_size;        % phi的维度
initialLearnRate = 5e-2;% 初始学习率
minLearnRate = 0;        % 最低学习率
num_epochs = 100;        % 训练轮数
L1 = 1000;                % 状态预测损失权重
L2 = 10.0;                % 高维状态预测损失权重
batchSize = 32768;


baseLayers = [
    imageInputLayer([feature_size, node_size, 1], 'Name','state_input')
    GraphConvolutionLayer(feature_size, hidden_size, adjMatrix, 'graph')
    reluLayer('Name', 'relu')
    functionLayer(@(X) dlarray(reshape(stripdims(X), [], size(X, 3)),'CB'), 'Name', 'reshape1', 'Formattable', true) 
    fullyConnectedLayer(output_size, 'Name', 'fc_phi')
];

% 创建初始网络
obj.Net = dlnetwork(baseLayers);

% 添加额外层
obj.Net = addLayers(obj.Net, featureInputLayer(control_size, 'Name', 'control_input'));  % 控制输入
obj.Net = addLayers(obj.Net, fullyConnectedLayer(feature_size*node_size + output_size, 'Name', 'A', 'BiasLearnRateFactor', 0)); % 无偏置线性层A
obj.Net = addLayers(obj.Net, fullyConnectedLayer(feature_size*node_size + output_size, 'Name', 'B', 'BiasLearnRateFactor', 0)); % 无偏置线性层B
obj.Net = addLayers(obj.Net, concatenationLayer(1, 2, 'Name', 'concat')); % 拼接state和phi
obj.Net = addLayers(obj.Net, additionLayer(2, 'Name','add')); % 相加A*Phi+B*control_input
obj.Net = addLayers(obj.Net, functionLayer(@(X) dlarray(reshape(stripdims(X), [], size(X, 4)),'CB'), 'Name', 'reshape', 'Formattable', true));


% 连接层
obj.Net = connectLayers(obj.Net, 'state_input', 'reshape');
obj.Net = connectLayers(obj.Net, 'reshape', 'concat/in1');
obj.Net = connectLayers(obj.Net, 'fc_phi', 'concat/in2');
obj.Net = connectLayers(obj.Net, 'concat', 'A');
obj.Net = connectLayers(obj.Net, 'control_input', 'B');
obj.Net = connectLayers(obj.Net, 'A', 'add/in1');
obj.Net = connectLayers(obj.Net, 'B', 'add/in2');


analyzeNetwork(obj.Net)
% 初始化网络
inputState = dlarray(rand(6,6,1,1),'SSCB');
inputControl = dlarray(rand(6,1),'CB');
obj.Net = initialize(obj.Net, inputState, inputControl);



