function test4regression
% testing function for regression problems (private)



% Values computed for trainSet and testSet with current and optimal parameters
%   
%   - error
%   - maxError
%           the maximum of the error
%   - maxRelativeError
%           the maximum of the relative error, that for each sample i is given by 
%           (t_i-o_i) /o_i
%           where t_i is the target and o_i the output of the system.

global dataSet dynamicSystem learning testing TestFigH

dataSet.testSet.forwardSteps=50;

supervisedNodesTrain=find(diag(dataSet.trainSet.maskMatrix));
supervisedNodesTest=find(diag(dataSet.testSet.maskMatrix));
supervisedNodesNumberTrain=size(supervisedNodesTrain,1);
supervisedNodesNumberTest=size(supervisedNodesTest,1);



%% Evaluating current parameters on trainset

% [x,currentTrainForwardState]=feval(dynamicSystem.config.forwardFunction,dataSet.testSet.forwardSteps,dynamicSystem.state,dataSet.trainSet...
%     ,dynamicSystem.parameters,dynamicSystem.config);
for nt = 1:dynamicSystem.ntrans
    [x{nt},currentTrainForwardState(nt)]=feval(dynamicSystem.config.forwardFunction,dataSet.testSet.forwardSteps,dynamicSystem.state{nt},'trainSet',0,nt);
end
% [x,currentTrainForwardState]=feval(dynamicSystem.config.forwardFunction,dataSet.testSet.forwardSteps,dynamicSystem.state,'trainSet',0);
% [testing.current.trainSet.error,currentTrainOutState]=feval(dynamicSystem.config.computeErrorFunction,x,dataSet.trainSet,...
%     dynamicSystem.parameters,dynamicSystem.config);
[testing.current.trainSet.error,currentTrainOutState]=feval(dynamicSystem.config.computeErrorFunction,'trainSet',x,0);
outError = currentTrainOutState.delta(:,supervisedNodesTrain);
targets = dataSet.trainSet.targets(:,supervisedNodesTrain);
% testing.current.trainSet.errorAll = outError;
testing.current.trainSet.relativeError = abs(outError ./ targets);
testing.current.trainSet.maxRelativeError=max(testing.current.trainSet.relativeError(:));
% FIXME: all following expressions are wrong since they do not consider the
% case that the targets are multi-dimensional. i.e. they should contain delta(:,supervisedNodesTrain)
testing.current.trainSet.acc5percent=size(find(abs(currentTrainOutState.delta(supervisedNodesTrain)) ./ dataSet.trainSet.targets(supervisedNodesTrain)<0.05),2)/supervisedNodesNumberTrain;
testing.current.trainSet.acc10percent=size(find(abs(currentTrainOutState.delta(supervisedNodesTrain)) ./ dataSet.trainSet.targets(supervisedNodesTrain)<0.1),2)/supervisedNodesNumberTrain;
testing.current.trainSet.maxError=max(abs(currentTrainOutState.delta(supervisedNodesTrain)));
testing.current.trainSet.x=x;
testing.current.trainSet.out=currentTrainOutState.outNetState.outs;
testing.current.trainSet.outState=currentTrainOutState;
testing.current.trainSet.forwardState=currentTrainForwardState;


%% Evaluating optimal parameters on trainset

% [x,trainForwardState]=feval(dynamicSystem.config.forwardFunction,dataSet.testSet.forwardSteps,dynamicSystem.state,dataSet.trainSet,...
%     learning.current.optimalParameters,dynamicSystem.config);
for i = 1:dynamicSystem.ntrans
    [x{i},trainForwardState(i)]=feval(dynamicSystem.config.forwardFunction,dataSet.testSet.forwardSteps,dynamicSystem.state{i},'trainSet',1,i);
end
% [x,trainForwardState]=feval(dynamicSystem.config.forwardFunction,dataSet.testSet.forwardSteps,dynamicSystem.state,'trainSet',1);
% [testing.optimal.trainSet.error,trainOutState]=feval(dynamicSystem.config.computeErrorFunction,x,dataSet.trainSet,...
%     learning.current.optimalParameters,dynamicSystem.config);
[testing.optimal.trainSet.error,trainOutState]=feval(dynamicSystem.config.computeErrorFunction,'trainSet',x,1);
outError = trainOutState.delta(:,supervisedNodesTrain);
targets = dataSet.trainSet.targets(:,supervisedNodesTrain);
% testing.optimal.trainSet.errorAll = outError;
testing.optimal.trainSet.relativeError = abs(outError ./ targets);
testing.optimal.trainSet.maxRelativeError=max(abs(trainOutState.delta(supervisedNodesTrain) ./ dataSet.trainSet.targets(supervisedNodesTrain)));
testing.optimal.trainSet.acc5percent=size(find(abs(trainOutState.delta(supervisedNodesTrain)) ./ dataSet.trainSet.targets(supervisedNodesTrain)<0.05),2)/supervisedNodesNumberTrain;
testing.optimal.trainSet.acc10percent=size(find(abs(trainOutState.delta(supervisedNodesTrain)) ./ dataSet.trainSet.targets(supervisedNodesTrain)<0.1),2)/supervisedNodesNumberTrain;
testing.optimal.trainSet.maxError=max(abs(trainOutState.delta(supervisedNodesTrain)));
testing.optimal.trainSet.x=x;
testing.optimal.trainSet.out=trainOutState.outNetState.outs;
testing.optimal.trainSet.outState=trainOutState;
testing.optimal.trainSet.forwardState=trainForwardState;


%% Evaluating current parameters on testset

% [x,currentTestForwardState]=feval(dynamicSystem.config.forwardFunction,dataSet.testSet.forwardSteps,...
%     zeros(dynamicSystem.config.nStates,dataSet.testSet.nNodes),dataSet.testSet,dynamicSystem.parameters,dynamicSystem.config);
% [x,currentTestForwardState]=feval(dynamicSystem.config.forwardFunction,dataSet.testSet.forwardSteps,...
%     zeros(dynamicSystem.config.nStates,dataSet.testSet.nNodes),dataSet.testSet,dynamicSystem.parameters,dynamicSystem.config);
for i = 1:dynamicSystem.ntrans
    [x{i},currentTestForwardState(i)]=feval(dynamicSystem.config.forwardFunction,dataSet.testSet.forwardSteps,sparse(dynamicSystem.config.nStates,dataSet.testSet.nNodes),'testSet',0,i);
end
% [x,currentTestForwardState]=feval(dynamicSystem.config.forwardFunction,dataSet.testSet.forwardSteps,sparse(dynamicSystem.config.nStates,dataSet.testSet.nNodes),'testSet',0);
% [testing.current.testSet.error,currentTestOutState]=feval(dynamicSystem.config.computeErrorFunction,x,dataSet.testSet,...
%     dynamicSystem.parameters,dynamicSystem.config);
[testing.current.testSet.error,currentTestOutState]=feval(dynamicSystem.config.computeErrorFunction,'testSet',x,0);
outError = currentTestOutState.delta(:,supervisedNodesTest);
targets = dataSet.testSet.targets(:,supervisedNodesTest);
testing.current.testSet.errorAll = outError;
testing.current.testSet.relativeError = abs(outError ./ targets);
testing.current.testSet.maxRelativeError=max(abs(currentTestOutState.delta(supervisedNodesTest) ./ dataSet.testSet.targets(supervisedNodesTest)));
testing.current.testSet.acc5percent=size(find(abs(currentTestOutState.delta(supervisedNodesTest)) ./ dataSet.testSet.targets(supervisedNodesTest)<0.05),2)/supervisedNodesNumberTest;
testing.current.testSet.acc10percent=size(find(abs(currentTestOutState.delta(supervisedNodesTest)) ./ dataSet.testSet.targets(supervisedNodesTest)<0.1),2)/supervisedNodesNumberTest;
testing.current.testSet.maxError=max(abs(currentTestOutState.delta(supervisedNodesTest)));
testing.current.testSet.x=x;
testing.current.testSet.out=currentTestOutState.outNetState.outs;
testing.current.testSet.outState=currentTestOutState;
testing.current.testSet.forwardState=currentTestForwardState;


%% Evaluating optimal parameters on testset

% [x,testForwardState]=feval(dynamicSystem.config.forwardFunction,dataSet.testSet.forwardSteps,
%     zeros(dynamicSystem.config.nStates,dataSet.testSet.nNodes),dataSet.testSet,...
%     learning.current.optimalParameters,dynamicSystem.config);
for nt = 1:dynamicSystem.ntrans
    [x{nt},testForwardState(nt)]=feval(dynamicSystem.config.forwardFunction,dataSet.testSet.forwardSteps,sparse(dynamicSystem.config.nStates,dataSet.testSet.nNodes),...
    'testSet',1,nt);
end
% [testing.optimal.testSet.error,testOutState]=feval(dynamicSystem.config.computeErrorFunction,x,dataSet.testSet,...
%     learning.current.optimalParameters,dynamicSystem.config);
[testing.optimal.testSet.error,testOutState]=feval(dynamicSystem.config.computeErrorFunction,'testSet',x,1);
outError = testOutState.delta(:,supervisedNodesTest);
targets = dataSet.testSet.targets(:,supervisedNodesTest);
testing.optimal.testSet.errorAll = outError;
testing.optimal.testSet.relativeError = abs(outError ./ targets);
testing.optimal.testSet.maxRelativeError=max(abs(testOutState.delta(supervisedNodesTest) ./ dataSet.testSet.targets(supervisedNodesTest)));
testing.optimal.testSet.acc5percent=size(find(abs(testOutState.delta(supervisedNodesTest)) ./ dataSet.testSet.targets(supervisedNodesTest)<0.05),2)/supervisedNodesNumberTest;
testing.optimal.testSet.acc10percent=size(find(abs(testOutState.delta(supervisedNodesTest)) ./ dataSet.testSet.targets(supervisedNodesTest)<0.1),2)/supervisedNodesNumberTest;
testing.optimal.testSet.maxError=max(abs(testOutState.delta(supervisedNodesTest)));
testing.optimal.testSet.x=x;
testing.optimal.testSet.out=testOutState.outNetState.outs;
testing.optimal.testSet.outState=testOutState;
testing.optimal.testSet.forwardState=testForwardState;


%% Displays results
global VisualMode
if VisualMode == 1
    TestFigH=DisplayTestR(testing.current.trainSet.error,testing.current.trainSet.acc5percent,testing.current.trainSet.acc10percent,...
        testing.optimal.trainSet.error,testing.optimal.trainSet.acc5percent,testing.optimal.trainSet.acc10percent,...
        testing.current.testSet.error,testing.current.testSet.acc5percent,testing.current.testSet.acc10percent,...
        testing.optimal.testSet.error,testing.optimal.testSet.acc5percent,testing.optimal.testSet.acc10percent);
elseif VisualMode == 2
    mssg(sprintf('\n\t\t\tTESTSET\t\t\t\t\tTRAINSET'));
    mssg(sprintf('--------------------------------------------------------------------------------------'));
    mssg([sprintf('\t\t| Error: \t\t') num2str(testing.optimal.testSet.error,'%10.5g') sprintf('\t\tError: \t\t\t') num2str(testing.optimal.trainSet.error,'%10.5g')]);
    mssg([sprintf('\t\t| maxError: \t\t') num2str(testing.optimal.testSet.maxError,'%10.5g') sprintf('\t\tmaxError: \t\t') num2str(testing.optimal.trainSet.maxError,'%10.5g')]);
    mssg([sprintf('OPTIMAL\t\t| maxRelativeError: \t') num2str(testing.optimal.testSet.maxRelativeError,'%10.5g') sprintf('\t\tmaxRelativeError: \t') num2str(testing.optimal.trainSet.maxRelativeError,'%10.5g')]);
    mssg([sprintf('\t\t| err < 0.05:\t\t') num2str(testing.optimal.testSet.acc5percent*100,'%4.2f') '%%' sprintf('\t\terr < 0.05:\t\t') num2str(testing.optimal.trainSet.acc5percent*100,'%4.2f') '%%']);
    mssg([sprintf('\t\t| err < 0.1:\t\t') num2str(testing.optimal.testSet.acc10percent*100,'%4.2f') '%%' sprintf('\t\terr < 0.1:\t\t') num2str(testing.optimal.trainSet.acc10percent*100,'%4.2f') '%%']);
    mssg(sprintf('\t\t|'));
    mssg([sprintf('\t\t| Error: \t\t') num2str(testing.current.testSet.error,'%10.5g') sprintf('\t\tError: \t\t\t') num2str(testing.current.trainSet.error,'%10.5g')]);
    mssg([sprintf('\t\t| maxError: \t\t') num2str(testing.current.testSet.maxError,'%10.5g') sprintf('\t\tmaxError: \t\t') num2str(testing.current.trainSet.maxError,'%10.5g')]);
    mssg([sprintf('CURRENT\t\t| maxRelativeError: \t') num2str(testing.current.testSet.maxRelativeError,'%10.5g') sprintf('\t\tmaxRelativeError: \t') num2str(testing.current.trainSet.maxRelativeError,'%10.5g')]);
    mssg([sprintf('\t\t| err < 0.05:\t\t') num2str(testing.current.testSet.acc5percent*100,'%4.2f') '%%' sprintf('\t\terr < 0.05:\t\t') num2str(testing.current.trainSet.acc5percent*100,'%4.2f') '%%']);
    mssg([sprintf('\t\t| err < 0.1:\t\t') num2str(testing.current.testSet.acc10percent*100,'%4.2f') '%%' sprintf('\t\terr < 0.1:\t\t') num2str(testing.current.trainSet.acc10percent*100,'%4.2f') '%%']);
    mssg(sprintf('--------------------------------------------------------------------------------------'));
end