function y = Generate_RandomInput(D,N,T,min_pressure,max_pressure)
    size_r = N*T;
    random_data = min_pressure + rand(D, size_r).*(max_pressure-min_pressure);
    y = zeros(D, N);
    for j=1:7
        y(j,:) = interp1(linspace(0, 1, T*N),  random_data(j,:), linspace(0, 1, N), 'spline');
    end
end

