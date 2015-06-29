function plotTrainingResults
global learning dynamicSystem

if isfield(learning.history, 'saturationCoefficient')
    figure('Name','Saturation Coefficients','NumberTitle','off');
    i=1;
    color=['b','r','k'];
    leg=cell(1,1);
    hold all
    tmp = learning.history.saturationCoefficient.outNet;
    plot([1:size(tmp,2)],tmp,color(i));
    leg{i} = 'outNet';
    i = i+1;
    N = numel(learning.history.saturationCoefficient.transitionNet);
    for nt = 1:N
        tmp = learning.history.saturationCoefficient.transitionNet{nt};
        plot([1:size(tmp,2)],tmp);
        leg{i} = ['transitionNet-' num2str(nt)];
        i = i+1;
    end
    hold off;
    title('Saturation Coefficients');
    legend(char(leg));
end

if isfield(learning.history, 'forwardItHistory')
    figure('Name','Forward and backward iterations','NumberTitle','off');
    plot([1:size(learning.history.forwardItHistory,2)],learning.history.forwardItHistory,'b');
    hold on
    plot([1:size(learning.history.backwardItHistory,2)],learning.history.backwardItHistory,'g');
    hold off
    title('Forward and backward iterations');
    legend('Forward iterations', 'Backward iterations');
end

if isfield(learning.history, 'stabilityCoefficientHistory')
    figure('Name','Stability Coefficient History','NumberTitle','off');
    hold all
    for nt = 1:dynamicSystem.ntrans
        plot([1:size(learning.history.stabilityCoefficientHistory{nt},2)],learning.history.stabilityCoefficientHistory{nt});
    end
    title('Stability Coefficient History');
end

if isfield(learning.history, 'jacobianHistory')
    figure('Name','Jacobian Error History','NumberTitle','off');
    hold all;
    for nt = 1:dynamicSystem.ntrans
        plot([1:size(learning.history.jacobianHistory{nt},2)],learning.history.jacobianHistory{nt});
    end
    title('Jacobian Error History');
end

if isfield(learning.history, 'jacobianHistoryComplete')
    figure('Name','Jacobian History','NumberTitle','off');
    hold all;
    for nt = 1:dynamicSystem.ntrans
        plot([1:size(learning.history.jacobianHistoryComplete{nt},2)],learning.history.jacobianHistoryComplete{nt});
    end
    title('Jacobian History');
end

if isfield(learning.history,'trainErrorHistory')
    figure('Name','Learning results','NumberTitle','off');
    plot([1:size(learning.history.trainErrorHistory,2)],learning.history.trainErrorHistory,'k');
    hold on
    t=[learning.config.stepsForValidation:learning.config.stepsForValidation:...
        learning.config.stepsForValidation*(size(learning.history.validationErrorHistory,2))];
    t(end)=learning.current.nSteps-1;
    plot(t, learning.history.validationErrorHistory,'r');
    hold off
    title('Learning results');
    legend('learning error', 'validation error');
end