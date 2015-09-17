% Mutagenesis example
clear all
load d
startSession
% Create a 10-fold cross validation data set
% makeMutagenicDataset
% global multidata
dataSet = d{1};
% Train the GNN by only 1 data set
Configure('GNN3.config')

global VisualMode
VisualMode = 2;
plotFlag = 1;
% analyzeError
drawnow

learn
% Test
test