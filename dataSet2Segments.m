function [dataSegments , dim] = dataSet2Segments(data , segmentNum)
    % 此函数将原始数据集划分成片段并类内随机排序，为了实现后面的交叉验证
    % 参数:
    % data：总sample行数 × feature列数。最后一列为label
    % segmentNum：交叉验证轮数，即分成多少个片段
    % 返回：
    % dataSegments：划分成segmentNum个片段的data：segmentNum×totalClass个cell
    % dim：feature维数
    
    totalClass = length(unique(data(: , end))); % 样本类别数
    dim = size(data , 2) - 1; % feature维数
    totalSample = size(data , 1);
    dataSegments = [];
    % labelSegments = [];
    for i = 1 : totalClass % i 是行，第几个类
        dataTemp{1 , i} = data(data(: , end) == i , 1 : end - 1);
        % 提取data中label列等于i的所有行，再去掉label列
        ithClassLength = size(dataTemp{1 , i} , 1); % 第i类data的行数
             
        index = randperm(ithClassLength); % 将第i类data顺序变乱序，index为[48 , 1 , 50 , 3 ,…… ]
        segSize = floor(ithClassLength/segmentNum); % 第i类里分成segSize个segment
        
        % 以下代码将每一列划分成segmentNum个
        for j = 1: segmentNum - 1 % 除了最后一个segment。j是列，第几个segment
            dataSegments{j , i} = dataTemp{1 , i}(index(segSize * (j - 1) + 1 : segSize * j) , :);
            % 5 × 3个cell，3类，每个cell里是10 × 4
        end
        dataSegments{j + 1 , i} = dataTemp{1 , i}(index(segSize * j + 1 : ithClassLength) , :);
        % 最后一个segment可能不够segSize个，因此是到ithClassLength
    end
end