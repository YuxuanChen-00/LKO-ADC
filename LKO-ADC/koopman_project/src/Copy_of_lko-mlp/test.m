state_size = 6;
hidden_size = 12;
output_size = 24;
control_size = 6;
time_step = 5;
net = lko_mlp_network(state_size, control_size, hidden_size, output_size, time_step);
net = net.Net;
analyzeNetwork(net)
fprintf('\n详细层索引列表:\n');
for i = 1:numel(net.Layers)
    % 显示层索引、层名称和层类型
    fprintf('Layer %2d: %-20s (%s)\n',...
        i,...
        net.Layers(i).Name,...
        class(net.Layers(i)));
end
