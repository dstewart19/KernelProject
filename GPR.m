%% ---------- GPR model(Version 2) ---------
% This file trains a GPR
[VLabels,EstimatedLabels,model,EstConfMatrices] = MyGPR(EstData,VData,EstLabels,VLabels,par)
%% extract the data 
if isempty(par)
    model = [];
else
    model = par;
end
fprintf('Loading Data ...\n');

if ~isempty(EstData)
fprintf('\nTraining GPR ...\n');

% Train the model and get parameters/hyperparameters (Use a linear basis to fit)
model = fitrgp(EstData,EstLabels,'Basis','linear', 'FitMethod','exact','KernelFunction','squaredexponential');
Parameters = model.Beta;
HyperParameter_LengthScale = model.KernelInformation.KernelParameters(1);
HyperParameter_SignalSTD = model.KernelInformation.KernelParameters(2);
HyperParameter_Sigma  = model.Sigma;
HpyerParameters = [HyperParameter_LengthScale,HyperParameter_SignalSTD,HyperParameter_Sigma];
EstimatedLabels = predict(model,VData);
fprintf('Validation Accuracy: %f\n', mean(double(EstLabels == VLabels)) * 100);
EstConfMatrices = confusionmat(VLabels,EstLabels);
else
   