%% 参数设置
min_pressure = [0,0,0,0,0,0,0]';
max_pressure = [5,5,5,5,5,5,2]';
t = 60;
fs = 20;
N = t*fs;
D = 7;
T = 0.04;
path = '..\Data\InputData\SonimInput_dataset_0.04_3.mat';
%% 单种信号同时激励
% 随机游走信号
maxStep = 2;  % 最大步长
probRise = 0.2;
probFall = 0.2;
probHold = 0.6;
signal_random_walk = Generate_RandomWalk(D, N, T, maxStep, probRise, probFall, probHold, min_pressure, max_pressure);

% 拉丁超立方采样信号
signal_lhs = Generate_LHS(D,N,T,min_pressure,max_pressure);

% 扫频信号
f0 = 0.005*rand;
f1 = 0.005 + 0.005*rand;
signal_chirp = Generate_Chirp(D,N,f0,f1,min_pressure,max_pressure);

% 多频正弦波信号
frequencies = [0,0.002;0.002,0.005;0.005,0.01]; % 每一行是一个频段
signal_multisine = Generate_MultiSine(D,N,frequencies,min_pressure,max_pressure);

% 随机信号插值
signal_random_input = Generate_RandomInput(D,N,T,min_pressure,max_pressure);

% PRBS激励信号
f0 = 0.008*rand;
f1 = 0.008 + 0.06*rand;
signal_PRBS = Generate_PRBS(D,N,f0,f1,min_pressure,max_pressure);
% %% 不同信号同时激励
% % 弯曲模块 多频正弦波+PRBS+随机信号；伸缩模块：扫频
% signal_jointed = zeros(D, N);
% num = N/3; %  每种信号组合的长度
% signal_jointed(7,:) = signal_chirp(7,:);
% 
% for i = 1:3
%     
%     current_segment = (i-1)*num+1:i*num;
%     % 生成前三个通道的数据
%     signal_jointed(mod(i,3)+1,current_segment) =  signal_multisine(mod(i,3)+1,current_segment);
%     signal_jointed(mod(i+1,3)+1,current_segment) =  signal_PRBS(mod(i+1,3)+1,current_segment);
%     signal_jointed(mod(i+2,3)+1,current_segment) =  signal_random_input(mod(i+2,3)+1,current_segment);
%     
%     % 生成后三个通道的数据
%     signal_jointed(mod(i,3)+4,current_segment) =  signal_multisine(mod(i,3)+4,current_segment);
%     signal_jointed(mod(i+1,3)+4,current_segment) =  signal_PRBS(mod(i+1,3)+4,current_segment);
%     signal_jointed(mod(i+2,3)+4,current_segment) =  signal_random_input(mod(i+2,3)+1,current_segment);
% 
% end

% %% 交替激励信号
% % 只激活第一个弯曲模块和伸缩模块
% signal3 = zeros(D, N);
% signal3(7,:) = chirp_signal(7,:);
% for i = 1:3
%     signal3(i,:) =  rand_walk_signal(i,:);
% end
% % 只激活第二个弯曲模块和伸缩模块
% signal4 = zeros(D, N);
% signal4(7,:) = rand_walk_signal(7,:);
% for i = 4:6
%     signal4(i,:) =  chirp_signal(i,:);
% end
%% 绘制采样信号
% signal = [signal1, signal2];
% for i = 1:7
%     plot(5800:5900,signal(i,5800:5900));
%     hold on;
% end

%% 将信号保存下来
save(path, "signal_random_walk", "signal_lhs", "signal_chirp", "signal_PRBS",...
    "signal_random_input","signal_multisine");