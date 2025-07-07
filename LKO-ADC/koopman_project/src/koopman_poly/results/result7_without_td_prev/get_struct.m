data = load('raw_results.mat');
sorted_results = data.sorted_results;
for i = 1:size(sorted_results, 1)
    current_dimension = sorted_results(i).dimension;
    current_delay = sorted_results(i).delay;
    sorted_results(i).dimension = current_dimension/current_delay;
end
save('full_results.mat', "sorted_results")

