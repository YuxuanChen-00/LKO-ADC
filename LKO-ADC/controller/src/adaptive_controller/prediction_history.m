classdef prediction_history
    properties
        history
        length
        data_num
    end
    methods
        function obj = prediction_history(state_num, length)
            obj.history = zeros(state_num, length);
            obj.length = length;
            obj.data_num = 0;
        end
        function obj = getdata(obj, data)
           if obj.data_num >= obj.length
               obj.history(:, 1:end-1) = obj.history(:, 2:end);
               obj.history(:, end) = data;
           else
               obj.data_num = obj.data_num + 1;
               obj.history(:, obj.data_num) = data;
           end
           % history = obj.history;
        end
    end
end

