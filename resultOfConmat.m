function [precision, recall, macroF1] = resultOfConmat(M)
    n = size(M, 1); % 对角元素数
    for i =1 : n % 
        if sum(M(:, i)) ~=0
            pre(i) = M(i, i) / sum(M(:, i)); % 该对角元素除以该列和，为查准率precision
        else
            pre(i) = 0;
        end
        if sum(M(i, :)) ~=0
            rec(i) = M(i, i) / sum(M(i, :)); % 该对角元素除以该行和，为召回率Recall
        else
            rec(i) = 0;
        end
    end
    precision = mean(pre);
    recall = mean(rec);
    macroF1 = (2*precision*recall) / (precision+recall);
end