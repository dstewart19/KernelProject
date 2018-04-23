function [X] = convTrainingData(XTrain,XLabels)
%function [X] = convTrainingData(XTrain)
% input  the training data and labels from Dr. Gader and convert to be used
% in the rest of the code where X is samples X features+1 where the final
% column is the true labels (1:Nclasses)
Labels = convClassLabels(XLabels);
XwLabels = cat(2,XTrain,Labels);
maxLabel = max(Labels);
X = [];

for class = 1:maxLabel
    classInd = find(Labels==class);
    X{class} = XwLabels(classInd,1:end-1);
end
end