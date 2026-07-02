function [res]= my_eval_y(y,Y)
% y is the prediction
% Y is the ground truth
[newIndx] = best_map(Y,y);
acc = mean(Y==newIndx);
nmi = mutual_info(Y,newIndx);
purity = pur_fun(Y,newIndx);
[AR,RI,MI,HI] = RandIndex(Y, newIndx);
[fscore,precision,recall] = compute_f(Y, newIndx);
CE = sum(Y(:) ~= y(:)) / length(Y);
res = [CE; acc; nmi; purity; AR; RI; MI; HI; fscore; precision; recall];