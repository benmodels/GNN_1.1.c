%function [jacobian,jacobianError]=neuralModelGetJacobian(dataset,p,forwardState,sys)
function [jacobian,jacobianError]=neuralModelGetJacobian(dataset,forwardState,n)
% Jacobian: (nStates * nNodes) x (nStates * nNodes)
global dataSet dynamicSystem learning

%xdim=dynamicSystem.config.nStates;
%tdj=sparse(xdim,dataSet.(dataset).nArcs*xdim);

tdj=sparse(dynamicSystem.config.nStates,dataSet.(dataset).nArcs*dynamicSystem.config.nStates);

for i=1:dynamicSystem.config.nStates
    eDelta=sparse(dynamicSystem.config.nStates,dataSet.(dataset).nArcs);
    % The dynamicSystem.config.transitionNet.backwardFunction
    % backpropagates the Jacobian from the output of F to its input. Here,
    % we inject 1 as the Jac at output i and get Jac from output i to all
    % inputs. Then repeat it for other outputs.
    eDelta(i,:)=1;
    
    if isempty(forwardState)
        [g,dj]= feval(dynamicSystem.config.transitionNet.backwardFunction,dynamicSystem.parameters.transitionNet(n),...
            learning.current.forwardState(n).transitionNetState,eDelta);
    else
        [g,dj]= feval(dynamicSystem.config.transitionNet.backwardFunction,dynamicSystem.parameters.transitionNet,...
            forwardState.transitionNetState,eDelta);
    end
    
    %%%%%%%%%%%%%%%%%%%%% OPTIMIZATION %%%%%%%%%%%%%%%%%%%%%%%%%%%
    %xdj=dj(1:xdim,:);
    %tdj(i,:)=xdj(:)';

    %% 1st version
    %tdj(i,:)=reshape(dj(1:xdim,:),1,xdim*dataset.nArcs);
    % assume we have only one instance of data. F block has nStates states in the
    % input and nStates in the output
    % row i gives the jacobian from x(i) in the output to all x in the
    % input. So, tdj(i,j) is the Jacobian from output i to input j
    % for more data instances, they are all stacked horizontally
    tdj(i,:)=reshape(dj(1:dynamicSystem.config.nStates,:),1,dynamicSystem.config.nStates*dataSet.(dataset).nArcs);
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
end

% Jacobian for the states at nodes
netJacobian=sparse(...
    dataSet.(dataset).neuralModel.toBlockMatrixRow,...
    dataSet.(dataset).neuralModel.toBlockMatrixColumn,...
    tdj(:),...
    dataSet.(dataset).nArcs*dynamicSystem.config.nStates, ...
    dataSet.(dataset).nArcs*dynamicSystem.config.nStates);

% Add the contribution of arcs to get jacobian for the nodes
% (nStates * nNodes) x (nStates * nNodes)
jacobian=dataSet.(dataset).neuralModel.childToArcMatrix' *...
    netJacobian * ...
    dataSet.(dataset).neuralModel.fatherToArcMatrix;

jacobianSum=sum(abs(jacobian));

jacobianError=(jacobianSum > dynamicSystem.config.jacobianThreshold) .* (jacobianSum - dynamicSystem.config.jacobianThreshold);
   