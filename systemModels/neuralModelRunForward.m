%function [x,state,i]=neuralModelRunForward(maxIt,x,dataset,p,sys,stopCoef)
function [x,state,i]=neuralModelRunForward(maxSteps,x,dataset,optimalParam,n)
%% This function just call forward a number "n" of times and compute the new state.

global dataSet dynamicSystem learning

if (isfield(dynamicSystem.config,'useLabelledEdges') && (dynamicSystem.config.useLabelledEdges==1))
    labels=[dataSet.(dataset).nodeLabels(:,dataSet.(dataset).neuralModel.childOfArc);dataSet.(dataset).nodeLabels(:,dataSet.(dataset).neuralModel.fatherOfArc);dataSet.(dataset).edgeLabels];
else
    labels=[dataSet.(dataset).nodeLabels(:,dataSet.(dataset).neuralModel.childOfArc);dataSet.(dataset).nodeLabels(:,dataSet.(dataset).neuralModel.fatherOfArc)];
end

%for i=1:maxIt
for i=1:maxSteps
    % Input: states of fathers, labels of children, labels of fathers
    % (labels of edges if available). Note that states of children is not
    % considered here. Number of rows is equal to the sum of these vector
    % sizes. Number of columns is equal to the total number of realtions,
    % i.e. total number of nonzero values in the connMatrix
    in=[x(:,dataSet.(dataset).neuralModel.fatherOfArc);labels];

    %[y,state.transitionNet]=feval(sys.transitionNet.forwardFunction,in,p.transitionNet);
    %s=dataset.neuralModel.childToArcMatrix' *y(:);
    %state.transitionNet=feval(sys.transitionNet.forwardFunction,in,'transitionNet');
    state.transitionNetState=feval(dynamicSystem.config.transitionNet.forwardFunction,in,'transitionNet',optimalParam,n);

    %s=dataset.neuralModel.childToArcMatrix' *state.transitionNet.outs(:);
    % state.transitionNetState correspond to the state values for each
    % "realation". The values for the relations corresponding to one node
    % should be merged (through childToArchMatrix) to give the state for
    % that particular node.
    s=dataSet.(dataset).neuralModel.childToArcMatrix' *state.transitionNetState.outs(:);
    nx=reshape(s,size(x));
    
    % Termination criteria
    stabCoef=(sum(sum(abs(x-nx)))) / sum(sum(abs(nx)));

    x=nx;
    if(stabCoef<learning.config.forwardStopCoefficient)
        break;
    end
end
