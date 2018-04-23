function [ bestParameters ] = GetBestParameters(confMatrices,Nf,parameters)
%function [ parameters ] = GetBestParameters(confMatrices,parameters,type)
%   Take in all confusion matrices for a given 2 class problem and return
%   the best parameters based on best classification of validation types
%   are 'SVM' 'RVM' and 'GPR' parameters will be a cell list for each of
%   the tests containing the best parameters out of the Nfold evaluations
%   computed
sumOffDiagonal = [];
for folds = 1:Nf
    sumOffDiagonal(folds) = sum(sum(confMatrices(folds).*~eye(size(confMatrices(folds)))));
end

[~,bestParametersInd] = min(sumOffDiagonal);
bestParameters = parameters{bestParametersInd};

end

