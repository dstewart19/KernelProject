function [classAccuracy,confusionMat] = TestMyClassifier(X,par,Alg)

XTrain = X(:,1:60); Labels = X(:,end);
[E,V,YE,YV] = randPart(XTrain,1);
E = cell2mat(E);
V = cell2mat(V);
YE = Labels(cell2mat(YE));
YV = Labels(cell2mat(YV));

switch Alg
    case 'SVM'
        [Ytrain,EstLabels,EstParameters,EstConfMatrices] = MySVM([],V,YE,YV,'Quadratic',par);
    case 'RVM'
        [Ytrain,EstLabels,EstParameters,EstConfMatrices] = MyRVM([],V,YE,YV,par);
    case 'GPR'
        [Ytrain,EstLabels,EstParameters,EstConfMatrices] = MyGPR([],V,YE,YV,par);
end


ValLabels = Ytrain';
confusionMat = confusionmat(ValLabels,EstLabels);
classAccuracy = (sum(diag(confusionMat))/sum(confusionMat(:)))*100;
end