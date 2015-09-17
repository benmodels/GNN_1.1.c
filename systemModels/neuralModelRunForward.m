%function [x,state,i]=neuralModelRunForward(maxIt,x,dataset,p,sys,stopCoef)
function [x,state,i]=neuralModelRunForward(maxSteps,x,dataset,optimalParam,n)
%% This function just call forward a number "n" of times and compute the new state.

global dataSet dynamicSystem learning
persistent childidx fatheridx intmp
r1 = size(x,1);
if learning.current.nSteps == 1
    intmp = struct();
end
if ~isfield(intmp,dataset)
    if (isfield(dynamicSystem.config,'useLabelledEdges') && (dynamicSystem.config.useLabelledEdges==1))
        labels=[dataSet.(dataset).nodeLabels(:,dataSet.(dataset).neuralModel.childOfArc);dataSet.(dataset).nodeLabels(:,dataSet.(dataset).neuralModel.fatherOfArc);dataSet.(dataset).edgeLabels];
        
    else
        fatheridx.(dataset) = dataSet.(dataset).neuralModel.fatherOfArc;
        childidx.(dataset) = dataSet.(dataset).neuralModel.childOfArc;
        labels=[dataSet.(dataset).nodeLabels(:,childidx.(dataset));dataSet.(dataset).nodeLabels(:,fatheridx.(dataset))];
        r2 = size(labels,1);
        c = size(labels,2);
        intmp.(dataset) = zeros(r1+r2, c);
        intmp.(dataset)(r1+1:r1+r2,:) = labels;
    end
end


%for i=1:maxIt
for i=1:maxSteps
    % Input: states of fathers, labels of children, labels of fathers
    % (labels of edges if available). Note that states of children is not
    % considered here. Number of rows is equal to the sum of these vector
    % sizes. Number of columns is equal to the total number of relations,
    % i.e. total number of nonzero values in the connMatrix
%     in=[x(:,dataSet.(dataset).neuralModel.fatherOfArc);labels];
    intmp.(dataset)(1:r1,:)=x(:,fatheridx.(dataset));

    %[y,state.transitionNet]=feval(sys.transitionNet.forwardFunction,in,p.transitionNet);
    %s=dataset.neuralModel.childToArcMatrix' *y(:);
    %state.transitionNet=feval(sys.transitionNet.forwardFunction,in,'transitionNet');
    state.transitionNetState=feval(dynamicSystem.config.transitionNet.forwardFunction,intmp.(dataset),'transitionNet',optimalParam,n);

    %s=dataset.neuralModel.childToArcMatrix' *state.transitionNet.outs(:);
    % state.transitionNetState correspond to the state values for each
    % "relation". The values for the relations corresponding to one node
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
