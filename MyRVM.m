function [VLabels,EstimatedLabels,model,EstConfMatrices] = MyRVM(EstData,VData,EstLabels,VLabels,par)
if ~isempty(EstData) && ~isempty(EstLabels)
    TrainingData = prtDataSetClass(EstData,EstLabels);
    TestingData = prtDataSetClass(VData,VLabels);
else
    TestingData = prtDataSetClass(VData,VLabels);
end
if isempty(par)
    classifier = prtClassRvm;
    fprintf('\nTraining RVM ...\n')
    classifier = classifier.train(TrainingData);
    fprintf('\n Validating RVM ...\n')
else
    classifier = par;
    fprintf('\n Testing RVM ...\n')
end

classified = run(classifier,TestingData);

%real labels
VLabels = classified.targets;

%estimated Labels
EstimatedLabels = round(classified.data);

model = classifier;
[EstConfMatrices,~] = prtScoreConfusionMatrix(EstimatedLabels,VLabels);
fprintf('Accuracy: %f\n', mean(double(EstimatedLabels == VLabels)) * 100);

end

