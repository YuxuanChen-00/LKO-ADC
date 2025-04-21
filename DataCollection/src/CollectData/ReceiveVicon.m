% 中断回调函数，用于接收vicon数据
function onemotion_data = ReceiveVicon(serialvicon,obj,event)
    global onemotion_data;
    data = fscanf(serialvicon);
    onemotion_data = str2num(data);
end