%% This function just calls backwark a given number of times and accumulate
%% gradient in dPar
%function [dPar,b,i]=neuralModelRunBackward(maxIt,x,dataset,p,delta,forwardState,sys,stopCoef)
function [dPar,b,I]=neuralModelRunBackward(delta,forwardState,maxIt)

global dataSet dynamicSystem learning

xdim=dynamicSystem.config.nStates;

%jacobian=feval(sys.forwardJacobianFunction,dataSet.trainSet,dynamicSystem.parameters,learning.current.forwardState,sys);
for j = 1:dynamicSystem.ntrans
    % Get Jacobian: (nStates * nNodes) x (nStates * nNodes)
    [learning.current.jacobian{j}, learning.current.jacobianErrors{j}] = feval(dynamicSystem.config.forwardJacobianFunction,'trainSet',forwardState,j);
    FblockIdx = [1:dynamicSystem.config.nStates]+ (j-1)*dynamicSystem.config.nStates;
    b=delta(FblockIdx,:);
    b = b(:);
    z=zeros(size(b));

    if isempty(maxIt)
        maxIt=learning.config.maxBackwardSteps;
    end

    i = 1;
    % Method 1: z = (I+A+A^2+A^3+...)b
    for i=1:maxIt
        z=z+b;
        b=learning.current.jacobian{j}' * b;
        stabCoefficient= sum(abs(b(:))) / sum(abs(z(:)));
        if(stabCoefficient < learning.config.backwardStopCoefficient) || (sum(abs(z(:))) == 0)
            break;
        end
    end

%     % Alternative: Method 2: Find z from equation 9
%     % z = (I-A)^-1 b ===> solve for (I-A)z = b
%     Id = speye(size(learning.current.jacobian{j}'));
%     z = full((Id-sparse(learning.current.jacobian{j}'))\b);
    
    I(1,j) = i;
    dPar.transitionNet(j)=feval(dynamicSystem.config.transitionNet.backwardFunction,...
        dynamicSystem.parameters.transitionNet(j),...
        learning.current.forwardState(j).transitionNetState,...
        reshape(z'*dataSet.trainSet.neuralModel.childToArcMatrix',xdim,dataSet.trainSet.nArcs));
end
