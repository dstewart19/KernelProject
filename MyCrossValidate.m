function [ValLabels,EstLabels,EstParameters,EstConfMatrices,ConfMatrix] = MyCrossValidate(X,Nf,Alg)
%function [Ytrain,EstParameters,EstConfMatrices,ConfMatrix] =
%MyCrossValidate(XTrain,Nf) cross validate the training data using Nf fold
%cross validation sets and return the trained labels (Ytrain),
%EstParameters, confusion matrices (EstConfMatrices), and overall confusion
%matrix ConfMatrix. X is samp X feat+1 where the final column is the
%normalized class labels samp X 1 where the value of each label is the
%class the sample is assigned to
Ytrain = []; EstParameters = []; EstConfMatrices = []; ConfMatrix = []; EstLabels =[];
XTrain = X(:,1:60); Labels = X(:,end);
[EE,VV,YEE,YVV] = randPart(XTrain,Nf);

% ClassType = 'Linear';   
ClassType = 'Quadratic';
% [sampSize,~] = size(E);
for tests = 1:Nf
    E = EE{tests}; V = VV{tests}; YE = Labels(YEE{tests}); YV = Labels(YVV{tests});
    
    switch Alg
        case 'RVM'
            [Ytrain{tests},EstLabels{tests},EstParameters{tests},EstConfMatrices(:,:,tests)] = MyRVM(E,V,YE,YV,[]);
        case 'SVM'
            [Ytrain{tests},EstLabels{tests},EstParameters{tests},EstConfMatrices(:,:,tests)] = MySVM(E,V,YE,YV,ClassType,[]);
        case 'GPR'
            [Ytrain{tests},EstLabels{tests},EstParameters{tests},EstConfMatrices(:,:,tests)] = MyGPR(E,V,YE,YV,[]);
    end
end
ValLabels = cell2mat(Ytrain');
EstLabels = cell2mat(EstLabels');
ConfMatrix = confusionmat(ValLabels,EstLabels);

