function [labels] = convClassLabels(oldLabels)
%function [labels] = convClassLabels(oldLabels)
%   convert 5 column labels that contain 1 for the class label in the class
%   column and -1 elsewhere to label matrix that contains only the number
%   of what class it contains
[~,ind] = find(oldLabels==1);
labels = ind;

end

