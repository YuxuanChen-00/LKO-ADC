%% 参数设置
min_pressure = [0,0,0,0,0,0,0]';
max_pressure = [5,5,5,5,5,5,5]';
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
segnum = 10;
% PRBS激励信号
f0 = 0.1;
f1 = 0.8;
signal_PRBS = Generate_SegPRBS(D,N,f0,f1,segnum,min_pressure,max_pressure);

% 绘制采样信号
signal = signal_PRBS;
for i = 1:7
    plot(1:N,signal(i,:));
    hold on;
end