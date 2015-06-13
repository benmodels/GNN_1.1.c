% Mutagenesis example
clear all
startSession
% Create a 10-fold cross validation data set
makeMutagenicDataset
global multidata
% Train the GNN by only 1 data set
dataSet = multidata(1)
Configure('GNN.config')
learn
% Test
test