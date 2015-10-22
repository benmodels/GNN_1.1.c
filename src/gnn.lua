require 'preprocess'
require 'nn'
require 'F_Module'


-- RETRIEVE DATA
local data_path = 'data/final_dataset.mat'
local dataSet = getDataset(data_path)

-- SET OTHER PARAMS
local nStates = 1
local useLabelledEdges = false
local maxSteps = 50
local forwardStopCoef = 1e-3
local delta, maxIter, forwardState, backwardStopCoef = nil, nil, nil, nil

-- For layer sizes
local numLabels = dataSet[1][1][6]:size(1)
local numHiddens = 3;

-- BUILD NET
local net = nn.Sequential()
local F = nn.F_Module(nStates, maxSteps, forwardStopCoef, delta, maxIter, forwardState, backwardStopCoef)

local F_internal = nn.Sequential()
F_internal:add( nn.Linear(nStates+2*numLabels, numHiddens) );
F_internal:add( nn.Tanh() )
F_internal:add( nn.Linear(numHiddens, nStates) );

F:add(F_internal)
net:add(F)

-- local mse = nn.MSECritereon()
-- local trainer = nn.StochasticGradientTrainer()


for _,d in pairs(dataSet) do
    yp = net:forward(d[1])
end