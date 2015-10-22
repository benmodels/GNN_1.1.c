local F_Module, parent = torch.class('nn.F_Module', 'nn.Container')

function F_Module:__init(nStates, maxSteps, forwardStopCoef, delta, maxIter, forwardState, backwardStopCoef)
    parent.__init(self)

    self.maxSteps, self.forwardStopCoef = maxSteps, forwardStopCoef

    self.delta, self.maxIter = delta, maxIter
    self.nStates, self.nArcs = nStates, nArcs
    self.forwardState, self.backwardStopCoef = forwardState, backwardStopCoef
end


--FORWARD
function F_Module:updateOutput(input)
    local childOfArc, fatherOfArc, childToArcMatrix, nNodes, nArcs, label, mask = unpack(input)

    local x = torch.zeros(self.nStates, nArcs)
    local labels = torch.cat(label:index(2, childOfArc),
                             label:index(2, fatherOfArc), 1)
    local input_vector = torch.cat(x,
                                   labels, 1)

    self.output:resizeAs(x):zero()

    for i=1,self.maxSteps do
        input_vector[{{1,self.nStates},{}}]:copy(self.output)

        local new_out = torch.zeros(x:size())
        for j=1,input_vector:size(2) do
            new_out[{{},j}] = self.modules[1]:forward(input_vector[{{},j}])
        end
        new_out = (new_out * childToArcMatrix):index(2, fatherOfArc)

        local stabCoef = relative_diff(self.output, new_out)
        self.output:copy(new_out)

        if stabCoef < self.forwardStopCoef then break end
    end

    return self.output
end

function relative_diff(first, second)
    local denom = second:abs():sum()
    if demon == 0 then return -1 end
    return (first-second):abs():sum() / denom
end


--BACKWARD
function F_Module:updateGradInput(input, gradOutput)
    self.gradInput:resizeAs(gradOutput):copy(gradOutput)
    childToArcMatrix = input[3]

    for i=1,nTrans do
        local jac = getJacobian(forwardState, i)

        local fblock_start, fblock_end = 1+(i-1)*nStates, i*nStates
        b = M2V( delta[{{fblock_start, fblock_end},{}}] )
        z = torch.zeros(b:size())

        for j=1,maxIter do
            z:add(b)
            b:cmul(jac)

            local stabCoef = relative_diff(b, z)
            if (not stabCoef) or (stabCoef < self.backwardStopCoef) then break end
        end

        z:cmul(childToArcMatrix):resize(nStates, nArcs)
        --dPar = backward(forwardState[i]), z) ??
    end

    return self.gradInput
end


function M2V(x)
    return torch.resize(x, x:nElement())
end