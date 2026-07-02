function L = getL(X, para)
    % each column is a data
    % para is construction

    if nargin < 2
        k = 20;
        sigma = 1;
    else
        if isfield(para, 'k') % weather k is member of para
            k = para.k;
        else
            k = 20;
        end;
        if isfield(para, 'sigma')
            sigma = para.sigma;
        else
            sigma = 1;
        end;
    end;
    
    [nFea, nSmp] = size(X); % 
    D = pdist2(X', X', 'Euclidean' );
    % calculate the distance of two sample, can be 'euclidean', 'minkowski' or 'mahalanobis'
    S = spalloc(nSmp,nSmp,20*nSmp); % spalloc allocate the memery space for sparse matrix
    % S has 20 non-zero elements
    
    [dumb idx] = sort(D, 2); % sort each row
    % each row, sort from small to large. idx is a matrix, contain the
    % index of original sample in the row.
    % idx become 1211*1211 matrix
    for i = 1 : nSmp % from 1 to 1211
        S(i,idx(i,2:k+1)) = 1;      % k is numLabel-1, namely 5   
        % W(1,idx(1,2:6))
    end
    S = (S+S')/2;
    
    D = diag(sum(S,2));
    L = D - S;
    
    
    
    