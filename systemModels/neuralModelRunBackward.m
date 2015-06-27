%% This function just calls backwark a given number of times and accumulate
%% gradient in dPar
%function [dPar,dX,i]=neuralModelRunBackward(maxIt,x,dataset,p,delta,forwardState,sys,stopCoef)
function [dPar,dX,i]=neuralModelRunBackward(delta,forwardState,maxIt)

global dataSet dynamicSystem learning

xdim=dynamicSystem.config.nStates;

%jacobian=feval(sys.forwardJacobianFunction,dataSet.trainSet,dynamicSystem.parameters,learning.current.forwardState,sys);
for i = 1:dynamicSystem.ntrans
    [learning.current.jacobian{i}, learning.current.jacobianErrors{i}] = feval(dynamicSystem.config.forwardJacobianFunction,'trainSet',forwardState,i);

    dX=delta(:);
    totDeltaX=zeros(size(dX));

    if isempty(maxIt)
        maxIt=learning.config.maxBackwardSteps;
    end
    for i=1:maxIt
        totDeltaX=totDeltaX+dX;
        dX=learning.current.jacobian{i}' * dX;
        stabCoefficient=sum(sum(abs(dX))) /sum(sum(abs(totDeltaX)));
        if(stabCoefficient < learning.config.backwardStopCoefficient) || (sum(sum(abs(totDeltaX))) == 0)
            break;
        end
    end


    dPar{i}.transitionNet=feval(dynamicSystem.config.transitionNet.backwardFunction,dynamicSystem.parameters.transitionNet{i},...
        learning.current.forwardState(i).transitionNetState,reshape(totDeltaX'*dataSet.trainSet.neuralModel.childToArcMatrix',xdim,dataSet.trainSet.nArcs));
end
