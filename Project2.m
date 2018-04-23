%% Project2

% This script will compute all the functions needed for Project2 
% April 17, 2018
% Dylan Stewart
Ytrain = []; EstParameters = []; EstConfMatrices = []; ConfMatrix = []; EstLabels = [];
% load training data
load('Proj2FeatVecsSet1.mat'); load('Proj2TargetOutputsSet1.mat');
Alg = {'SVM' 'RVM' 'GPR'};
Algs = length(Alg);
% Choose how many folds to cross validate
Nf = 5;
XTrain = Proj2FeatVecsSet1; XLabels = Proj2TargetOutputsSet1;
Nfeat = size(XTrain,2);
x = convTrainingData(XTrain,XLabels);

%% Training and Cross Validation
TestSets = nchoosek(1:5,2);
NumTests = nchoosek(5,2);

% TO TRAIN WITH NEW DATA UNCOMMENT THIS
% xTrain = cell(size(x));
% for classes = 1:length(x)
%     xTrain{classes} = x{classes}(1:0.8*length(XTrain),:);
%     x{classes}(1:0.8*length(XTrain,:) = [];
% end
% for algorithms = 1:Algs
%     BestParameters = [];
%     for tests = 1:NumTests
%         
%         X1 = cat(2,xTrain{TestSets(tests,1)},ones(length(xTrain{TestSets(tests,1)}),1));
%         X2 = cat(2,xTrain{TestSets(tests,2)},zeros(length(xTrain{TestSets(tests,2)}),1));
%         X = cat(1,X1,X2);
%         fprintf('Training Classifier ...\n')
%         [Ytrain{tests},EstLabels{tests},EstParameters{tests},EstConfMatrices{tests},ConfMatrix{tests}] = MyCrossValidate(X,Nf,Alg{algorithms});
%         [BestParameters{tests}] = GetBestParameters(EstConfMatrices{tests},Nf,EstParameters{tests});   
%     end
%     save(strcat(Alg{algorithms},'_Results.mat'),'Ytrain','EstLabels','EstParameters','EstConfMatrices','ConfMatrix','BestParameters');
% end

%% Testing
classAccuracy = []; confusionMat = [];
for algorithms = 1:Algs
    file = strcat(Alg{algorithms},'_Results.mat');
    load(file)
    
    for tests = 1:NumTests
        X1 = cat(2,x{TestSets(tests,1)},ones(length(x{TestSets(tests,1)}),1));
        X2 = cat(2,x{TestSets(tests,2)},zeros(length(x{TestSets(tests,2)}),1));
        X = cat(1,X1,X2);
        fprintf('Testing Classifier ...\n')
        [classAccuracy(tests),confusionMat{tests}] = TestMyClassifier(X,BestParameters{tests},Alg{algorithms});
    end
     save(strcat(Alg{algorithms},'_TestResults.mat'),'classAccuracy','confusionMat');
end

        


