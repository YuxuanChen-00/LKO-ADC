classdef MatMul_To_GemmLayer1029 < nnet.layer.Layer & nnet.layer.Formattable
    % A custom layer auto-generated while importing an ONNX network.
    
    %#codegen
    %#ok<*PROPLC>
    %#ok<*NBRAK>
    %#ok<*INUSL>
    %#ok<*VARARG>
    
    properties (Learnable)
        C_0_bias
        C_0_weight
        base_mlp_0_bias
        base_mlp_0_weight
        base_mlp_2_bias
        base_mlp_2_weight
        onnx__MatMul_129
        onnx__MatMul_130
    end
    
    properties
        ONNXParams         % An ONNXParameters object containing parameters used by this layer.
    end
    
    methods
        function this = MatMul_To_GemmLayer1029(name, onnxParams)
            this.Name = name;
            this.NumInputs = 5;
            this.NumOutputs = 3;
            this.OutputNames = {'phi_current', 'phi_pred', 'state_pred'};
            this.ONNXParams = onnxParams;
            this.C_0_bias = onnxParams.Learnables.C_0_bias;
            this.C_0_weight = onnxParams.Learnables.C_0_weight;
            this.base_mlp_0_bias = onnxParams.Learnables.base_mlp_0_bias;
            this.base_mlp_0_weight = onnxParams.Learnables.base_mlp_0_weight;
            this.base_mlp_2_bias = onnxParams.Learnables.base_mlp_2_bias;
            this.base_mlp_2_weight = onnxParams.Learnables.base_mlp_2_weight;
            this.onnx__MatMul_129 = onnxParams.Learnables.onnx__MatMul_129;
            this.onnx__MatMul_130 = onnxParams.Learnables.onnx__MatMul_130;
        end
        
        function [phi_current, phi_pred, state_pred] = predict(this, x_base_lstm_base__7, state_current, control_current, state_currentNumDims, control_currentNumDims)
            if isdlarray(x_base_lstm_base__7)
                x_base_lstm_base__7 = stripdims(x_base_lstm_base__7);
            end
            if isdlarray(state_current)
                state_current = stripdims(state_current);
            end
            if isdlarray(control_current)
                control_current = stripdims(control_current);
            end
            x_base_lstm_base__7NumDims = 3;
            state_currentNumDims = numel(state_currentNumDims);
            control_currentNumDims = numel(control_currentNumDims);
            onnxParams = this.ONNXParams;
            onnxParams.Learnables.C_0_bias = this.C_0_bias;
            onnxParams.Learnables.C_0_weight = this.C_0_weight;
            onnxParams.Learnables.base_mlp_0_bias = this.base_mlp_0_bias;
            onnxParams.Learnables.base_mlp_0_weight = this.base_mlp_0_weight;
            onnxParams.Learnables.base_mlp_2_bias = this.base_mlp_2_bias;
            onnxParams.Learnables.base_mlp_2_weight = this.base_mlp_2_weight;
            onnxParams.Learnables.onnx__MatMul_129 = this.onnx__MatMul_129;
            onnxParams.Learnables.onnx__MatMul_130 = this.onnx__MatMul_130;
            [phi_current, phi_pred, state_pred, phi_currentNumDims, phi_predNumDims, state_predNumDims] = MatMul_To_GemmFcn(x_base_lstm_base__7, state_current, control_current, x_base_lstm_base__7NumDims, state_currentNumDims, control_currentNumDims, onnxParams, 'Training', false, ...
                'InputDataPermutation', {[3 2 1], ['as-is'], ['as-is'], ['as-is'], ['as-is'], ['as-is']}, ...
                'OutputDataPermutation', {['as-is'], ['as-is'], ['as-is'], ['as-is'], ['as-is'], ['as-is']});
            if any(cellfun(@(A)~isnumeric(A) && ~islogical(A), {phi_current, phi_pred, state_pred}))
                fprintf('Runtime error in network. At least one output of custom layer ''%s'' is a non-numeric, non-logical value.\n', 'MatMul_To_GemmLayer1029');
                error(message('nnet_cnn_onnx:onnx:BadCustomLayerRuntimeOutput', 'MatMul_To_GemmLayer1029'));
            end
            phi_current = dlarray(single(phi_current), repmat('U', 1, max(2, phi_currentNumDims)));
            phi_pred = dlarray(single(phi_pred), repmat('U', 1, max(2, phi_predNumDims)));
            state_pred = dlarray(single(state_pred), repmat('U', 1, max(2, state_predNumDims)));
            if ~coder.target('MATLAB')
                phi_current = extractdata(phi_current);
                phi_pred = extractdata(phi_pred);
                state_pred = extractdata(state_pred);
            end
        end
        
        function [phi_current, phi_pred, state_pred] = forward(this, x_base_lstm_base__7, state_current, control_current, state_currentNumDims, control_currentNumDims)
            if isdlarray(x_base_lstm_base__7)
                x_base_lstm_base__7 = stripdims(x_base_lstm_base__7);
            end
            if isdlarray(state_current)
                state_current = stripdims(state_current);
            end
            if isdlarray(control_current)
                control_current = stripdims(control_current);
            end
            x_base_lstm_base__7NumDims = 3;
            state_currentNumDims = numel(state_currentNumDims);
            control_currentNumDims = numel(control_currentNumDims);
            onnxParams = this.ONNXParams;
            onnxParams.Learnables.C_0_bias = this.C_0_bias;
            onnxParams.Learnables.C_0_weight = this.C_0_weight;
            onnxParams.Learnables.base_mlp_0_bias = this.base_mlp_0_bias;
            onnxParams.Learnables.base_mlp_0_weight = this.base_mlp_0_weight;
            onnxParams.Learnables.base_mlp_2_bias = this.base_mlp_2_bias;
            onnxParams.Learnables.base_mlp_2_weight = this.base_mlp_2_weight;
            onnxParams.Learnables.onnx__MatMul_129 = this.onnx__MatMul_129;
            onnxParams.Learnables.onnx__MatMul_130 = this.onnx__MatMul_130;
            [phi_current, phi_pred, state_pred, phi_currentNumDims, phi_predNumDims, state_predNumDims] = MatMul_To_GemmFcn(x_base_lstm_base__7, state_current, control_current, x_base_lstm_base__7NumDims, state_currentNumDims, control_currentNumDims, onnxParams, 'Training', true, ...
                'InputDataPermutation', {[3 2 1], ['as-is'], ['as-is'], ['as-is'], ['as-is'], ['as-is']}, ...
                'OutputDataPermutation', {['as-is'], ['as-is'], ['as-is'], ['as-is'], ['as-is'], ['as-is']});
            if any(cellfun(@(A)~isnumeric(A) && ~islogical(A), {phi_current, phi_pred, state_pred}))
                fprintf('Runtime error in network. At least one output of custom layer ''%s'' is a non-numeric, non-logical value.\n', 'MatMul_To_GemmLayer1029');
                error(message('nnet_cnn_onnx:onnx:BadCustomLayerRuntimeOutput', 'MatMul_To_GemmLayer1029'));
            end
            phi_current = dlarray(single(phi_current), repmat('U', 1, max(2, phi_currentNumDims)));
            phi_pred = dlarray(single(phi_pred), repmat('U', 1, max(2, phi_predNumDims)));
            state_pred = dlarray(single(state_pred), repmat('U', 1, max(2, state_predNumDims)));
            if ~coder.target('MATLAB')
                phi_current = extractdata(phi_current);
                phi_pred = extractdata(phi_pred);
                state_pred = extractdata(state_pred);
            end
        end
    end
end

function [phi_current, phi_pred, state_pred, phi_currentNumDims, phi_predNumDims, state_predNumDims, state] = MatMul_To_GemmFcn(x_base_lstm_base__7, state_current, control_current, x_base_lstm_base__7NumDims, state_currentNumDims, control_currentNumDims, params, varargin)
%MATMUL_TO_GEMMFCN Function implementing an imported ONNX network.
%
% THIS FILE WAS AUTO-GENERATED BY importONNXFunction.
% ONNX Operator Set Version: 13
%
% Variable names in this function are taken from the original ONNX file.
%
% [PHI_CURRENT, PHI_PRED, STATE_PRED] = MatMul_To_GemmFcn(X_BASE_LSTM_BASE__7, STATE_CURRENT, CONTROL_CURRENT, PARAMS)
%			- Evaluates the imported ONNX network MATMUL_TO_GEMMFCN with input(s)
%			X_BASE_LSTM_BASE__7, STATE_CURRENT, CONTROL_CURRENT and the imported network parameters in PARAMS. Returns
%			network output(s) in PHI_CURRENT, PHI_PRED, STATE_PRED.
%
% [PHI_CURRENT, PHI_PRED, STATE_PRED, STATE] = MatMul_To_GemmFcn(X_BASE_LSTM_BASE__7, STATE_CURRENT, CONTROL_CURRENT, PARAMS)
%			- Additionally returns state variables in STATE. When training,
%			use this form and set TRAINING to true.
%
% [__] = MatMul_To_GemmFcn(X_BASE_LSTM_BASE__7, STATE_CURRENT, CONTROL_CURRENT, PARAMS, 'NAME1', VAL1, 'NAME2', VAL2, ...)
%			- Specifies additional name-value pairs described below:
%
% 'Training'
% 			Boolean indicating whether the network is being evaluated for
%			prediction or training. If TRAINING is true, state variables
%			will be updated.
%
% 'InputDataPermutation'
%			'auto' - Automatically attempt to determine the permutation
%			 between the dimensions of the input data and the dimensions of
%			the ONNX model input. For example, the permutation from HWCN
%			(MATLAB standard) to NCHW (ONNX standard) uses the vector
%			[4 3 1 2]. See the documentation for IMPORTONNXFUNCTION for
%			more information about automatic permutation.
%
%			'none' - Input(s) are passed in the ONNX model format. See 'Inputs'.
%
%			numeric vector - The permutation vector describing the
%			transformation between input data dimensions and the expected
%			ONNX input dimensions.%
%			cell array - If the network has multiple inputs, each cell
%			contains 'auto', 'none', or a numeric vector.
%
% 'OutputDataPermutation'
%			'auto' - Automatically attempt to determine the permutation
%			between the dimensions of the output and a conventional MATLAB
%			dimension ordering. For example, the permutation from NC (ONNX
%			standard) to CN (MATLAB standard) uses the vector [2 1]. See
%			the documentation for IMPORTONNXFUNCTION for more information
%			about automatic permutation.
%
%			'none' - Return output(s) as given by the ONNX model. See 'Outputs'.
%
%			numeric vector - The permutation vector describing the
%			transformation between the ONNX output dimensions and the
%			desired output dimensions.%
%			cell array - If the network has multiple outputs, each cell
%			contains 'auto', 'none' or a numeric vector.
%
% Inputs:
% -------
% X_BASE_LSTM_BASE__7, STATE_CURRENT, CONTROL_CURRENT
%			- Input(s) to the ONNX network.
%			  The input size(s) expected by the ONNX file are:
%				  X_BASE_LSTM_BASE__7:		[Unknown, Unknown, Unknown]				Type: FLOAT
%				  STATE_CURRENT:		[batch_size, 6]				Type: FLOAT
%				  CONTROL_CURRENT:		[batch_size, 6]				Type: FLOAT
%			  By default, the function will try to permute the input(s)
%			  into this dimension ordering. If the default is incorrect,
%			  use the 'InputDataPermutation' argument to control the
%			  permutation.
%
%
% PARAMS	- Network parameters returned by 'importONNXFunction'.
%
%
% Outputs:
% --------
% PHI_CURRENT, PHI_PRED, STATE_PRED
%			- Output(s) of the ONNX network.
%			  Without permutation, the size(s) of the outputs are:
%				  PHI_CURRENT:		[Eluphi_current_dim_0, 225]				Type: FLOAT
%				  PHI_PRED:		[Addphi_pred_dim_0, 225]				Type: FLOAT
%				  STATE_PRED:		[Addphi_pred_dim_0, 6]				Type: FLOAT
%			  By default, the function will try to permute the output(s)
%			  from this dimension ordering into a conventional MATLAB
%			  ordering. If the default is incorrect, use the
%			  'OutputDataPermutation' argument to control the permutation.
%
% STATE		- (Optional) State variables. When TRAINING is true, these will
% 			  have been updated from the original values in PARAMS.State.
%
%
%  See also importONNXFunction

% Preprocess the input data and arguments:
[x_base_lstm_base__7, state_current, control_current, Training, outputDataPerms, anyDlarrayInputs] = preprocessInput(x_base_lstm_base__7, state_current, control_current, params, varargin{:});
% Put all variables into a single struct to implement dynamic scoping:
[Vars, NumDims] = packageVariables(params, {'x_base_lstm_base__7', 'state_current', 'control_current'}, {x_base_lstm_base__7, state_current, control_current}, [x_base_lstm_base__7NumDims state_currentNumDims control_currentNumDims]);
% Call the top-level graph function:
[phi_current, phi_pred, state_pred, phi_currentNumDims, phi_predNumDims, state_predNumDims, state] = MatMul_To_GemmGraph1016(x_base_lstm_base__7, state_current, control_current, NumDims.x_base_lstm_base__7, NumDims.state_current, NumDims.control_current, Vars, NumDims, Training, params.State);
% Postprocess the output data
[phi_current, phi_pred, state_pred] = postprocessOutput(phi_current, phi_pred, state_pred, outputDataPerms, anyDlarrayInputs, Training, varargin{:});
end

function [phi_current, phi_pred, state_pred, phi_currentNumDims1026, phi_predNumDims1027, state_predNumDims1028, state] = MatMul_To_GemmGraph1016(x_base_lstm_base__7, state_current, control_current, x_base_lstm_base__7NumDims1023, state_currentNumDims1024, control_currentNumDims1025, Vars, NumDims, Training, state)
% Function implementing the graph 'MatMul_To_GemmGraph1016'
% Update Vars and NumDims from the graph's formal input parameters. Note that state variables are already in Vars.
Vars.x_base_lstm_base__7 = x_base_lstm_base__7;
NumDims.x_base_lstm_base__7 = x_base_lstm_base__7NumDims1023;
Vars.state_current = state_current;
NumDims.state_current = state_currentNumDims1024;
Vars.control_current = control_current;
NumDims.control_current = control_currentNumDims1025;

% Execute the operators:
% Gather:
[Vars.x_Gather_output_0, NumDims.x_Gather_output_0] = onnxGather(Vars.x_base_lstm_base__7, Vars.x_Constant_output_0, 0, NumDims.x_base_lstm_base__7, NumDims.x_Constant_output_0);

% Concat:
[Vars.x_Concat_1_output_0, NumDims.x_Concat_1_output_0] = onnxConcat(1, {Vars.state_current, Vars.x_Gather_output_0}, [NumDims.state_current, NumDims.x_Gather_output_0]);

% Gemm:
[A, B, C, alpha, beta, NumDims.x_base_mlp_base_mlp_] = prepareGemmArgs(Vars.x_Concat_1_output_0, Vars.base_mlp_0_weight, Vars.base_mlp_0_bias, Vars.Gemmalpha1017, Vars.Gemmbeta1018, 0, 1, NumDims.base_mlp_0_bias);
Vars.x_base_mlp_base_mlp_ = alpha*B*A + beta*C;

% Elu:
[Vars.x_base_mlp_base_ml_1, NumDims.x_base_mlp_base_ml_1] = onnxElu(Vars.x_base_mlp_base_mlp_, 1.000000, NumDims.x_base_mlp_base_mlp_);

% Gemm:
[A, B, C, alpha, beta, NumDims.x_base_mlp_base_ml_2] = prepareGemmArgs(Vars.x_base_mlp_base_ml_1, Vars.base_mlp_2_weight, Vars.base_mlp_2_bias, Vars.Gemmalpha1019, Vars.Gemmbeta1020, 0, 1, NumDims.base_mlp_2_bias);
Vars.x_base_mlp_base_ml_2 = alpha*B*A + beta*C;

% Elu:
[Vars.phi_current, NumDims.phi_current] = onnxElu(Vars.x_base_mlp_base_ml_2, 1.000000, NumDims.x_base_mlp_base_ml_2);

% MatMul:
[Vars.x_A_MatMul_output_0, NumDims.x_A_MatMul_output_0] = onnxMatMul(Vars.phi_current, Vars.onnx__MatMul_129, NumDims.phi_current, NumDims.onnx__MatMul_129);

% MatMul:
[Vars.x_B_MatMul_output_0, NumDims.x_B_MatMul_output_0] = onnxMatMul(Vars.control_current, Vars.onnx__MatMul_130, NumDims.control_current, NumDims.onnx__MatMul_130);

% Add:
Vars.phi_pred = Vars.x_A_MatMul_output_0 + Vars.x_B_MatMul_output_0;
NumDims.phi_pred = max(NumDims.x_A_MatMul_output_0, NumDims.x_B_MatMul_output_0);

% Gemm:
[A, B, C, alpha, beta, NumDims.state_pred] = prepareGemmArgs(Vars.phi_pred, Vars.C_0_weight, Vars.C_0_bias, Vars.Gemmalpha1021, Vars.Gemmbeta1022, 0, 1, NumDims.C_0_bias);
Vars.state_pred = alpha*B*A + beta*C;

% Set graph output arguments from Vars and NumDims:
phi_current = Vars.phi_current;
phi_currentNumDims1026 = NumDims.phi_current;
phi_pred = Vars.phi_pred;
phi_predNumDims1027 = NumDims.phi_pred;
state_pred = Vars.state_pred;
state_predNumDims1028 = NumDims.state_pred;
% Set output state from Vars:
state = updateStruct(state, Vars);
end

function [inputDataPerms, outputDataPerms, Training] = parseInputs(x_base_lstm_base__7, state_current, control_current, numDataOutputs, params, varargin)
% Function to validate inputs to MatMul_To_GemmFcn:
p = inputParser;
isValidArrayInput = @(x)isnumeric(x) || isstring(x);
isValidONNXParameters = @(x)isa(x, 'ONNXParameters');
addRequired(p, 'x_base_lstm_base__7', isValidArrayInput);
addRequired(p, 'state_current', isValidArrayInput);
addRequired(p, 'control_current', isValidArrayInput);
addRequired(p, 'params', isValidONNXParameters);
addParameter(p, 'InputDataPermutation', 'auto');
addParameter(p, 'OutputDataPermutation', 'auto');
addParameter(p, 'Training', false);
parse(p, x_base_lstm_base__7, state_current, control_current, params, varargin{:});
inputDataPerms = p.Results.InputDataPermutation;
outputDataPerms = p.Results.OutputDataPermutation;
Training = p.Results.Training;
if isnumeric(inputDataPerms)
    inputDataPerms = {inputDataPerms};
end
if isstring(inputDataPerms) && isscalar(inputDataPerms) || ischar(inputDataPerms)
    inputDataPerms = repmat({inputDataPerms},1,3);
end
if isnumeric(outputDataPerms)
    outputDataPerms = {outputDataPerms};
end
if isstring(outputDataPerms) && isscalar(outputDataPerms) || ischar(outputDataPerms)
    outputDataPerms = repmat({outputDataPerms},1,numDataOutputs);
end
end

function [x_base_lstm_base__7, state_current, control_current, Training, outputDataPerms, anyDlarrayInputs] = preprocessInput(x_base_lstm_base__7, state_current, control_current, params, varargin)
% Parse input arguments
[inputDataPerms, outputDataPerms, Training] = parseInputs(x_base_lstm_base__7, state_current, control_current, 3, params, varargin{:});
anyDlarrayInputs = any(cellfun(@(x)isa(x, 'dlarray'), {x_base_lstm_base__7, state_current, control_current}));
% Make the input variables into unlabelled dlarrays:
x_base_lstm_base__7 = makeUnlabeledDlarray(x_base_lstm_base__7);
state_current = makeUnlabeledDlarray(state_current);
control_current = makeUnlabeledDlarray(control_current);
% Permute inputs if requested:
x_base_lstm_base__7 = permuteInputVar(x_base_lstm_base__7, inputDataPerms{1}, 3);
state_current = permuteInputVar(state_current, inputDataPerms{2}, 2);
control_current = permuteInputVar(control_current, inputDataPerms{3}, 2);
end

function [phi_current, phi_pred, state_pred] = postprocessOutput(phi_current, phi_pred, state_pred, outputDataPerms, anyDlarrayInputs, Training, varargin)
% Set output type:
if ~anyDlarrayInputs && ~Training
    if isdlarray(phi_current)
        phi_current = extractdata(phi_current);
    end
    if isdlarray(phi_pred)
        phi_pred = extractdata(phi_pred);
    end
    if isdlarray(state_pred)
        state_pred = extractdata(state_pred);
    end
end
% Permute outputs if requested:
phi_current = permuteOutputVar(phi_current, outputDataPerms{1}, 2);
phi_pred = permuteOutputVar(phi_pred, outputDataPerms{2}, 2);
state_pred = permuteOutputVar(state_pred, outputDataPerms{3}, 2);
end


%% dlarray functions implementing ONNX operators:

function [Y, numDimsY] = onnxConcat(ONNXAxis, XCell, numDimsXArray)
% Concatentation that treats all empties the same. Necessary because
% dlarray.cat does not allow, for example, cat(1, 1x1, 1x0) because the
% second dimension sizes do not match.
numDimsY = numDimsXArray(1);
XCell(cellfun(@isempty, XCell)) = [];
if isempty(XCell)
    Y = dlarray([]);
else
    if ONNXAxis<0
        ONNXAxis = ONNXAxis + numDimsY;
    end
    DLTAxis = numDimsY - ONNXAxis;
    Y = cat(DLTAxis, XCell{:});
end
end

function [X, numDimsX] = onnxElu(X, alpha, numDimsX)
% Implements the ONNX Elu operator
X(X<=0) = alpha*(exp(X(X<=0))-1);
end

function [Y, numDimsY] = onnxGather(X, ONNXIdx, ONNXAxis, numDimsX, numDimsIdx)
% Function implementing the ONNX Gather operator

% In ONNX, 'Gather' first indexes into dimension ONNXAxis of data, using
% the contents of ONNXIdx as the indices. Then, it reshapes the ONNXAxis
% into the shape of ONNXIdx.
%   Example 1:
% Suppose data has shape [2 3 4 5], ONNXIdx has shape [6 7], and axis=1.
% The result has shape [2 6 7 4 5].
%   Example 2:
% Suppose data has shape [2 3 4 5], ONNXIdx has shape [6], and axis=1.
% The result has shape [2 6 4 5].
%   Example 3:
% Suppose data has shape [2 3 4 5], ONNXIdx has shape [] (a scalar), and axis=1.
% The result has shape [2 4 5].
%
% Since we're using reverse indexing relative to ONNX, in this function
% data and ONNXIdx both have reversed dimension ordering.
numDimsY = numDimsIdx + (numDimsX - 1);
if isempty(X)
    Y = X;
    return;
end
% (1) First, do the subsref part of Gather
if ONNXAxis<0
    ONNXAxis = ONNXAxis + numDimsX;                                 % Axis can be negative. Convert it to its positive equivalent.
end
dltAxis = numDimsX - ONNXAxis;                                      % Convert axis to DLT. ONNXAxis is origin 0 and we index from the end
ONNXIdx(ONNXIdx<0) = ONNXIdx(ONNXIdx<0) + size(X, dltAxis);         % ONNXIdx can have negative components. Make them positive.
dltIdx  = extractdata(ONNXIdx) + 1;                                 % ONNXIdx is origin-0 in ONNX, so add 1 to get dltIdx
% Use subsref to index into data
Indices.subs = repmat({':'}, 1, numDimsX);
Indices.subs{dltAxis} = dltIdx(:);                                  % Index as a column to ensure the output is 1-D in the indexed dimension (for now).
Indices.type = '()';
Y = subsref(X, Indices);
% (2) Now do the reshaping part of Gather
shape = size(Y, 1:numDimsX);
if numDimsIdx == 0
    % Delete the indexed dimension
    shape(dltAxis) = [];
elseif numDimsIdx > 1
    % Reshape the indexed dimension into the shape of ONNXIdx
    shape = [shape(1:dltAxis-1) size(ONNXIdx, 1:numDimsIdx) shape(dltAxis+1:end)];
end
% Extend the shape to 2D so it's valid MATLAB
if numel(shape) < 2
    shape = [shape ones(1,2-numel(shape))];
end
Y = reshape(Y, shape);
end

function [D, numDimsD] = onnxMatMul(A, B, numDimsA, numDimsB)
% Implements the ONNX MatMul operator.

% If B is 1-D, temporarily extend it to a row vector
if numDimsB==1
    B = B(:)';
end
maxNumDims = max(numDimsA, numDimsB);
numDimsD = maxNumDims;
if maxNumDims > 2
    % Removes dlarray formats if only one of the input dlarrays is formatted.
    if sum([isempty(dims(A)), isempty(dims(B))]) == 1
        D = pagemtimes(stripdims(B), stripdims(A));
    else
        %computes matrix product of corresponding pages of input arrays A and
        %B.
        D = pagemtimes(B, A);
    end
else
    D = B * A;
    if numDimsA==1 || numDimsB==1
        D = D(:);
        numDimsD = 1;
    end
end
end

function [A, B, C, alpha, beta, numDimsY] = prepareGemmArgs(A, B, C, alpha, beta, transA, transB, numDimsC)
% Prepares arguments for implementing the ONNX Gemm operator
if transA
    A = A';
end
if transB
    B = B';
end
if numDimsC < 2
    C = C(:);   % C can be broadcast to [N M]. Make C a col vector ([N 1])
end
numDimsY = 2;
% Y=B*A because we want (AB)'=B'A', and B and A are already transposed.
end

%% Utility functions:

function s = appendStructs(varargin)
% s = appendStructs(s1, s2,...). Assign all fields in s1, s2,... into s.
if isempty(varargin)
    s = struct;
else
    s = varargin{1};
    for i = 2:numel(varargin)
        fromstr = varargin{i};
        fs = fieldnames(fromstr);
        for j = 1:numel(fs)
            s.(fs{j}) = fromstr.(fs{j});
        end
    end
end
end

function checkInputSize(inputShape, expectedShape, inputName)

if numel(expectedShape)==0
    % The input is a scalar
    if ~isequal(inputShape, [1 1])
        inputSizeStr = makeSizeString(inputShape);
        error(message('nnet_cnn_onnx:onnx:InputNeedsResize',inputName, "[1,1]", inputSizeStr));
    end
elseif numel(expectedShape)==1
    % The input is a vector
    if ~shapeIsColumnVector(inputShape) || ~iSizesMatch({inputShape(1)}, expectedShape)
        expectedShape{2} = 1;
        expectedSizeStr = makeSizeString(expectedShape);
        inputSizeStr = makeSizeString(inputShape);
        error(message('nnet_cnn_onnx:onnx:InputNeedsResize',inputName, expectedSizeStr, inputSizeStr));
    end
else
    % The input has 2 dimensions or more
    
    % The input dimensions have been reversed; flip them back to compare to the
    % expected ONNX shape.
    inputShape = fliplr(inputShape);
    
    % If the expected shape has fewer dims than the input shape, error.
    if numel(expectedShape) < numel(inputShape)
        expectedSizeStr = strjoin(["[", strjoin(string(expectedShape), ","), "]"], "");
        error(message('nnet_cnn_onnx:onnx:InputHasGreaterNDims', inputName, expectedSizeStr));
    end
    
    % Prepad the input shape with trailing ones up to the number of elements in
    % expectedShape
    inputShape = num2cell([ones(1, numel(expectedShape) - length(inputShape)) inputShape]);
    
    % Find the number of variable size dimensions in the expected shape
    numVariableInputs = sum(cellfun(@(x) isa(x, 'char') || isa(x, 'string'), expectedShape));
    
    % Find the number of input dimensions that are not in the expected shape
    % and cannot be represented by a variable dimension
    nonMatchingInputDims = setdiff(string(inputShape), string(expectedShape));
    numNonMatchingInputDims  = numel(nonMatchingInputDims) - numVariableInputs;
    
    expectedSizeStr = makeSizeString(expectedShape);
    inputSizeStr = makeSizeString(inputShape);
    if numNonMatchingInputDims == 0 && ~iSizesMatch(inputShape, expectedShape)
        % The actual and expected input dimensions match, but in
        % a different order. The input needs to be permuted.
        error(message('nnet_cnn_onnx:onnx:InputNeedsPermute',inputName, expectedSizeStr, inputSizeStr));
    elseif numNonMatchingInputDims > 0
        % The actual and expected input sizes do not match.
        error(message('nnet_cnn_onnx:onnx:InputNeedsResize',inputName, expectedSizeStr, inputSizeStr));
    end
end
end

function doesMatch = iSizesMatch(inputShape, expectedShape)
% Check whether the input and expected shapes match, in order.
% Size elements match if (1) the elements are equal, or (2) the expected
% size element is a variable (represented by a character vector or string)
doesMatch = true;
for i=1:numel(inputShape)
    if ~(isequal(inputShape{i},expectedShape{i}) || ischar(expectedShape{i}) || isstring(expectedShape{i}))
        doesMatch = false;
        return
    end
end
end

function sizeStr = makeSizeString(shape)
sizeStr = strjoin(["[", strjoin(string(shape), ","), "]"], "");
end

function isVec = shapeIsColumnVector(shape)
if numel(shape) == 2 && shape(2) == 1
    isVec = true;
else
    isVec = false;
end
end
function X = makeUnlabeledDlarray(X)
% Make numeric X into an unlabelled dlarray
if isa(X, 'dlarray')
    X = stripdims(X);
elseif isnumeric(X)
    if isinteger(X)
        % Make ints double so they can combine with anything without
        % reducing precision
        X = double(X);
    end
    X = dlarray(X);
end
end

function [Vars, NumDims] = packageVariables(params, inputNames, inputValues, inputNumDims)
% inputNames, inputValues are cell arrays. inputRanks is a numeric vector.
Vars = appendStructs(params.Learnables, params.Nonlearnables, params.State);
NumDims = params.NumDimensions;
% Add graph inputs
for i = 1:numel(inputNames)
    Vars.(inputNames{i}) = inputValues{i};
    NumDims.(inputNames{i}) = inputNumDims(i);
end
end

function X = permuteInputVar(X, userDataPerm, onnxNDims)
% Returns reverse-ONNX ordering
if onnxNDims == 0
    return;
elseif onnxNDims == 1 && isvector(X)
    X = X(:);
    return;
elseif isnumeric(userDataPerm)
    % Permute into reverse ONNX ordering
    if numel(userDataPerm) ~= onnxNDims
        error(message('nnet_cnn_onnx:onnx:InputPermutationSize', numel(userDataPerm), onnxNDims));
    end
    perm = fliplr(userDataPerm);
elseif isequal(userDataPerm, 'auto') && onnxNDims == 4
    % Permute MATLAB HWCN to reverse onnx (WHCN)
    perm = [2 1 3 4];
elseif isequal(userDataPerm, 'as-is')
    % Do not permute the input
    perm = 1:ndims(X);
else
    % userDataPerm is either 'none' or 'auto' with no default, which means
    % it's already in onnx ordering, so just make it reverse onnx
    perm = max(2,onnxNDims):-1:1;
end
X = permute(X, perm);
end

function Y = permuteOutputVar(Y, userDataPerm, onnxNDims)
switch onnxNDims
    case 0
        perm = [];
    case 1
        if isnumeric(userDataPerm)
            % Use the user's permutation because Y is a column vector which
            % already matches ONNX.
            perm = userDataPerm;
        elseif isequal(userDataPerm, 'auto')
            % Treat the 1D onnx vector as a 2D column and transpose it
            perm = [2 1];
        else
            % userDataPerm is 'none'. Leave Y alone because it already
            % matches onnx.
            perm = [];
        end
    otherwise
        % ndims >= 2
        if isnumeric(userDataPerm)
            % Use the inverse of the user's permutation. This is not just the
            % flip of the permutation vector.
            perm = onnxNDims + 1 - userDataPerm;
        elseif isequal(userDataPerm, 'auto')
            if onnxNDims == 2
                % Permute reverse ONNX CN to DLT CN (do nothing)
                perm = [];
            elseif onnxNDims == 4
                % Permute reverse onnx (WHCN) to MATLAB HWCN
                perm = [2 1 3 4];
            else
                % User wants the output in ONNX ordering, so just reverse it from
                % reverse onnx
                perm = onnxNDims:-1:1;
            end
        elseif isequal(userDataPerm, 'as-is')
            % Do not permute the input
            perm = 1:ndims(Y);
        else
            % userDataPerm is 'none', so just make it reverse onnx
            perm = onnxNDims:-1:1;
        end
end
if ~isempty(perm)
    Y = permute(Y, perm);
end
end

function s = updateStruct(s, t)
% Set all existing fields in s from fields in t, ignoring extra fields in t.
for name = transpose(fieldnames(s))
    s.(name{1}) = t.(name{1});
end
end
