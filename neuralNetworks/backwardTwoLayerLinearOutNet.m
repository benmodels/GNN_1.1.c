%function [gradient,dInputs]=backwardTwoLayerLinearOutNet(net,netState,delta,saturationControl,networkType)
function [gradient,dInputs]=backwardTwoLayerLinearOutNet(net,netState,delta)
% delta: grad w.r.t the output 
% net: structure that has the weights in it
% netState: structure that as the inputs to the input layer and hidden
% layer
% gradient: a structure that will have the gradient of cost w.r.t each element
% dInputs: gradient of cost w.r.t to the inputs of net
global dynamicSystem comparisonNet

gradient.weights2=delta*netState.hiddens';

gradient.bias2=sum(delta,2);
% z--> [tanh()] -->y
dyh_dzh = (1-netState.hiddens.*netState.hiddens);
dnet1=(net.weights2'*delta) .* dyh_dzh;

if dynamicSystem.config.useSaturationControl
    absval=abs(netState.hiddens)-dynamicSystem.config.saturationThreshold;
    absval(absval<0)=0;
    dnet1 = dnet1 + dynamicSystem.config.saturationCoeff.*absval.*sign(netState.hiddens);
end
gradient.weights1=dnet1*netState.inputs';

gradient.bias1=sum(dnet1,2);
dInputs=net.weights1'*dnet1;
