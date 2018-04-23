%% ---------- GPR model(Version 2) ---------
% This file trains a GPR
function [VLabels,EstimatedLabels,model,EstConfMatrices] = MyGPR(EstData,VData,EstLabels,VLabels,par)
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
    EstimatedLabels = round(predict(model,VData));
    EstimatedLabels(EstimatedLabels>1)=1;
    EstimatedLabels(EstimatedLabels<0)=0;
    fprintf('Validation Accuracy: %f\n', mean(double(EstimatedLabels == VLabels)) * 100);
    EstConfMatrices = confusionmat(VLabels,EstimatedLabels);
else
    p = round(predict(model, VData));
    fprintf('Testing Accuracy: %f\n', mean(double(p == VLabels)) * 100);
    EstimatedLabels = p;
    EstimatedLabels(EstimatedLabels>1)=1;
    EstimatedLabels(EstimatedLabels<0)=0;
    EstConfMatrices = confusionmat(VLabels,p);
end
end
