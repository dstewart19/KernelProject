function [E,V,YE,YV] = randPart(XTrain,Nf)
%function [E,V,YE,YV] = randPart(XTrain,Nf) randomly partitions XTrain into N
%folds for cross validation and returns YE and YV as the indices of each
%set of observations in a sample X fold dimensional matrix
%   input training data XTrain and the number of folds for cross validation
%   Nf, the data is randomly partitioned and original indices of each
%   sample are stored in YE and YV where YE is the indices of the
%   estimation sets in a sample X fold dimensional matrix and YV is similar
%   for the validation sets. E and V are samples X features X folds
%   dimensional
[rows,cols] = size(XTrain);
idx = randperm(rows);
sampSize = rows/(Nf*2);
E = zeros(sampSize,cols,Nf);
YE = zeros(sampSize,Nf);
V = zeros(size(E));
YV = zeros(size(YE));
indices = 1:sampSize:rows;
% fill in the Estimation Sets and Validation Sets
for fold = 1:Nf
    ii = indices(fold):indices(fold+1)-1;
E(:,:,fold) = XTrain(idx(ii),:);
YE(:,fold) = idx(ii);
    ij = indices(fold+Nf-1):indices(fold+Nf)-1;
V(:,:,fold) = XTrain(idx(ij),:);
YV(:,fold) = idx(ij);
end
EE = permute(E,[3 1 2]);
VV = permute(V,[3 1 2]);
YE = YE';
YV = YV';
E = []; V = [];
for cellCol= 1:Nf
E{1,cellCol} = reshape(EE(cellCol,:,:),[sampSize cols]);
V{1,cellCol} = reshape(VV(cellCol,:,:),[sampSize cols]);
end
YE = num2cell(YE,2);
YV = num2cell(YV,2);
end

