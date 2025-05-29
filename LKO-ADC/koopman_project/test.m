inputSize = 12;
numHiddenUnits = 100;
numClasses = 9;

layers = [ ...
    sequenceInputLayer(inputSize)
    lstmLayer(numHiddenUnits,OutputMode="sequence")
    flattenLayer()
    fullyConnectedLayer(numClasses)
    softmaxLayer];

analyzeNetwork(layers)

