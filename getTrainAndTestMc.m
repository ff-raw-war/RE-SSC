function [trainSet , testSet , labelMatTrain, labelMatTest, labelVecTrain, labelVecTest] = getTrainAndTestMc(dataSet , ithCycle , totalCycle)

    
    totalClass = size(dataSet , 2);   
    trainSet = [];
    testSet = [];
    labelMatTrain = [];
    labelMatTest = [];
    labelVecTrain = [];
    labelVecTest = [];
    trainCells = cell(1 , totalClass); 
    trainLabelCells = cell(1 , totalClass); 
    testCells = cell(1 , totalClass);
    testLabelCells = cell(1 , totalClass);
    
    lenTrain = 0; % 
    widTrain = 0;
    lenTest = 0;
    widTest = 0;


    for i = 1 : totalCycle
        for j = 1 : totalClass
            if i ~= ithCycle 
                lenTrain = lenTrain + size(dataSet{i , j} , 1);
                widTrain = widTrain + size(dataSet{i , j} , 2);
                trainCells{1 , j} = [trainCells{1 , j} ; dataSet{i , j}(: , :)];
                trainLabelCells{1 , j}= [trainLabelCells{1 , j} ; ones(size(dataSet{i , j} , 1) , 1)];
            else
                lenTest = lenTest + size(dataSet{i , j} , 1);
                widTest = widTest + size(dataSet{i , j} , 2);
                testCells{1 , j} = [testCells{1 , j} ; dataSet{i , j}(: , :)];
                testLabelCells{1 , j}= [testLabelCells{1 , j} ; ones(size(dataSet{i , j} , 1) , 1)];
            end
        end
    end

    for i = 1 : totalClass 
        labelMatTrain = blkdiag(labelMatTrain , trainLabelCells{1 , i}); 
        labelVecTrain = [labelVecTrain ; trainLabelCells{1, i}*i];
        labelMatTest = blkdiag(labelMatTest , testLabelCells{1 , i});
        labelVecTest = [labelVecTest ; testLabelCells{1, i}*i];
        trainSet = [trainSet ; trainCells{1 , i}];
        testSet = [testSet ; testCells{1 , i}];
    end
    
    
end