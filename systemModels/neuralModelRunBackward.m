%% This function just calls backwark a given number of times and accumulate
%% gradient in dPar
%function [dPar,dX,i]=neuralModelRunBackward(maxIt,x,dataset,p,delta,forwardState,sys,stopCoef)
function [dPar,dX,I]=neuralModelRunBackward(delta,forwardState,maxIt)

global dataSet dynamicSystem learning

xdim=dynamicSystem.config.nStates;

%jacobian=feval(sys.forwardJacobianFunction,dataSet.trainSet,dynamicSystem.parameters,learning.current.forwardState,sys);
for j = 1:dynamicSystem.ntrans
    [learning.current.jacobian{j}, learning.current.jacobianErrors{j}] = feval(dynamicSystem.config.forwardJacobianFunction,'trainSet',forwardState,j);
    idx = [1:dynamicSystem.config.nStates]+ (j-1)*dynamicSystem.config.nStates;
    dX=delta(idx,:);
    dX = dX(:);
    totDeltaX=zeros(size(dX));

    if isempty(maxIt)
        maxIt=learning.config.maxBackwardSteps;
    end
    for i=1:maxIt
        totDeltaX=totDeltaX+dX;
        dX=learning.current.jacobian{j}' * dX;
        stabCoefficient=sum(sum(abs(dX))) /sum(sum(abs(totDeltaX)));
        if(stabCoefficient < learning.config.backwardStopCoefficient) || (sum(sum(abs(totDeltaX))) == 0)
            break;
        end
    end

    I(1,j) = i;
    dPar.transitionNet(j)=feval(dynamicSystem.config.transitionNet.backwardFunction,dynamicSystem.parameters.transitionNet(j),...
        learning.current.forwardState(j).transitionNetState,reshape(totDeltaX'*dataSet.trainSet.neuralModel.childToArcMatrix',xdim,dataSet.trainSet.nArcs));
end
