function [VLabels,EstimatedLabels,model,EstConfMatrices] = MySVM(EstData,VData,EstLabels,VLabels,ClassType,par)

%% CAP6610 - Machine Learning
%  Project 2 | Classification
%
%  Instructions
%  ------------
% 
%  Contains two examples of running an SVM
%  1. Linear SVM on dataset1
%  2. Non-linear SVM on dataset2
%  
%  This script uses the following helper scripts;
%     gaussianKernel.m - used for non-linear classification using SVM
%     plotData.m - visual 2D data points
%     svmTrain.m - train SVM
%     svmPredict.m - given the trained model and new datapoint, predict Y
%     visualizeBoundary.m - visualize linear boundaries
%     visualizeBoundaryLinear.m - visualize linear boundaries
%

%% Initialization
if isempty(par)
    model = [];
else
    model = par;
end

% switch ClassType
%     case 'Linear'
        
%% =============== Loading and Visualizing Data ================

fprintf('Loading Data ...\n')

%% ==================== Training Linear SVM ====================
%  The following code will train a linear SVM on the dataset and plot the
%  decision boundary learned.
% if ~isempty(EstData)
%     fprintf('\nTraining Linear SVM ...\n')
%     
%     % Parameter C is a positive value that control the penalty for
%     % misclassified training examples
%     C = 1;
%     model = svmTrain(EstData, EstLabels, C, @linearKernel, 1e-3, 20);
%     %get validation accuracy
%     p = svmPredict(model, VData);
%     fprintf('Validation Accuracy: %f\n', mean(double(p == VLabels)) * 100);
%     EstimatedLabels = p;
%     EstConfMatrices = confusionmat(VLabels,p);
% else
%     p = svmPredict(model, VData);
%     fprintf('Validation Accuracy: %f\n', mean(double(p == VLabels)) * 100);
%     EstimatedLabels = p;
%     EstConfMatrices = confusionmat(VLabels,p);
% end

%% ========== Training non-linear SVM ==========
%  Train the SVM classifier using a Gaussian Kernel
%
%     case 'Quadratic'
        if ~isempty(EstData)
            fprintf('\nTraining SVM with RBF Kernel (this may take 1 to 2 minutes) ...\n');
            
            % SVM Parameters
            % sigma used by the gaussian kernel
            C = 1; sigma = 0.1;
            
            % Tolerance and max_passes set lower here so that the code will run
            % faster. However, in practice, we can run the training to
            % convergence.
            model= svmTrain(EstData, EstLabels, C, @(x1, x2) gaussianKernel(x1, x2, sigma));
            % get validation accuracy
            p = svmPredict(model,VData);
            fprintf('Validation Accuracy: %f\n', mean(double(p == VLabels)) * 100);
            EstimatedLabels = p;
            EstConfMatrices = confusionmat(VLabels,p);
        else
            p = svmPredict(model, VData);
            fprintf('Testing Accuracy: %f\n', mean(double(p == VLabels)) * 100);
            EstimatedLabels = p;
            EstConfMatrices = confusionmat(VLabels,p);
        end
end
