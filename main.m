clc;                        % Clear the command window
clear all, close all;       % Remove all variables from memory and close all open figure windows
start0 = tic;               % Start a global timer to measure the total execution time of the script
addpath('./eval/');         % Add the evaluation metrics folder to the MATLAB search path
addpath('./picShow/');      % Add the visualization/picture display folder to the search path
addpath('../data/');        % Add the dataset folder to the search path

% Define the search space for regularization/trade-off parameters
para = [10^2; 10^1; 1 ; 10^-1; 10^-2];          % Main parameter grid (alpha, beta, gamma, lambda)
EP = [10^2; 10^0; 10^-2; 10^-4; 10^-6];         % Epsilon parameter grid (ep1, ep2)
epsilon = 10^-1;                                % A constant threshold value used in the training process
maxIter = 10;                                   % Maximum number of iterations for the optimization algorithm
totalCycle = 2;                                 % Number of times to run each parameter combination to average results

% Dataset configuration
dataSetName = 'ORL_32x32';                      % Name of the .mat file to be loaded
load(dataSetName);                              % Load the dataset (expected to contain 'fea' and 'gnd')
method = 'Seventh 1.4.1';                       % Label identifying the current algorithm version

X = double(fea);                                % Convert feature data to double precision for calculations
label = gnd;                                    % Assign the ground truth labels to 'label'
totalClass = numel(unique(label));              % Count the number of unique classes in the dataset

% File logging setup
postName = strcat(dataSetName , '.txt');        % Create a filename suffix based on the dataset
saveFileName = strcat('.\report\',method,'_', postName); % Define the full path for the log file
warning off;                                    % Suppress MATLAB warnings during execution
fid = fopen(saveFileName,'a');                  % Open (or create) the log file in 'append' mode
fprintf('Method:%s, Dataset:%s, Time:%s\n', method, dataSetName, datetime);     % Print header to console
fprintf(fid, 'Method:%s, Dataset:%s, Time:%s\n', method, dataSetName, datetime); % Print header to file
string = '******';                              % Define a separator string
fprintf(string);                                % Print separator to console
fprintf(fid, string);                           % Print separator to file

result = [];                                    % Initialize an empty array to store metric results

[N,D] = size(X);                                % Get the number of samples (N) and feature dimensions (D)
minCE = 100;                                    % Initialize minimum clustering error (not used in this snippet)
bestAcc = 0;                                    % Initialize the best Accuracy found so far
bestNmi = 0;                                    % Initialize the best Normalized Mutual Information
bestPurity = 0;                                 % Initialize the best Purity
bestTimePerTrain = 100000;                      % Initialize the best training time with a very large value

% Format string for displaying parameters
paraString = 'alpha:%.2f, beta:%.2f, gamma:%.2f, lambda:%.2f, ep1:%.6f, ep2:%.6f'; 
cishu = 0;                                      % Counter for successful parameter updates

% Start 6-level nested loops for Grid Search over all parameter combinations
for ithAlpha = 1 : size(para)
    for ithBeta = 1 : size(para)
        for ithGamma = 1 : size(para)
            for ithLambda = 1:size(para)
                for ithEp1 = 1:size(EP)
                    for ithEp2 = 1:size(EP)
                        % Assign current iteration parameters
                        ep1 = EP(ithEp1);
                        ep2 = EP(ithEp2);
                        alpha = para(ithAlpha);
                        beta = para(ithBeta);
                        gamma = para(ithGamma);
                        lambda = para(ithLambda);
                        
                        paraCurrent = [alpha, beta, gamma, lambda, ep1, ep2]; % Store current combo in a vector
    
                        % Log current parameter set to console and file
                        fprintf(strcat('Current Parameters:',paraString,' \n'), paraCurrent);
                        fprintf(fid,strcat('Current Parameters:',paraString,'  \n'), paraCurrent);
    
                        result = []; % Reset results array for the current parameter combination
                        
                        % Repeat the experiment for the current parameters to handle stochasticity
                        for i = 1 : totalCycle 
                            start1 = tic;       % Start timer for a single training run
                            if gpuDeviceCount >= 1
                                X = gpuArray(X); % Move data to GPU memory if a GPU is available
                            end
                            
                            % Core training function: returns Projection (P), Error (E), Weights (W), and Coefficients (C)
                            % zscore(X') normalizes the data to have zero mean and unit variance
                            [P, E, W, C] = train(zscore(X'), alpha, beta, gamma, lambda, ep1, ep2, totalClass, epsilon, maxIter);
                            
                            timePerTrain = toc(start1); % Stop timer and record training duration

			                if gpuDeviceCount >= 1
                                C = gather(C);   % Move the coefficient matrix back from GPU to CPU memory
                            end
                            
                            % Build an adjacency matrix from C using a thresholding function
                            CKSym = BuildAdjacency(thrC(C, 1)); 
                            % Perform Spectral Clustering on the adjacency matrix to get predicted labels
                            labelPredict = SpectralClustering(CKSym, totalClass); 

                            % Evaluate clustering performance against ground truth
                            [res] = my_eval_y1(labelPredict, label); 
                            % Append metrics: [ACC, NMI, Purity, F-score, Adj-Rand, Recall, Time]
                            result = [result; [res(2) res(3) res(4) res(9) res(5) res(11) timePerTrain]]; 
                        end
                        
                        % Calculate summary statistics for the cycles
                        result(totalCycle+1, :) = mean(result);            % Store mean of all metrics
                        result(totalCycle+2, :) = std(result(1:totalCycle, :)); % Store standard deviation
                        result(totalCycle+3, :) = max(result);             % Store best run in the cycle
     
                        % Extract average values for reporting
                        avgAcc = result(totalCycle+1 , 1); 
                        avgNmi = result(totalCycle+1 , 2);
                        avgPurity = result(totalCycle+1 , 3);
                        avgF1 = result(totalCycle+1 , 4);
                        avgAR = result(totalCycle+1 , 5);
                        avgRecall = result(totalCycle+1, 6);
                        avgTimePerTrain = result(totalCycle+1, 7);
    
                        % Extract standard deviation values for reporting
                        stdAcc = result(totalCycle+2 , 1);
                        stdNmi = result(totalCycle+2 , 2);
                        stdPurity = result(totalCycle+2 , 3);
                        stdF1 = result(totalCycle+2 , 4);
                        stdAR = result(totalCycle+2 , 5);
                        stdRecall = result(totalCycle+2, 6);
                        stdTimePerTrain = result(totalCycle+2, 7);
                        
                        % Update the global best result if current average Accuracy is higher
                        if avgAcc > bestAcc
                            paraBest = [alpha, beta, gamma, lambda, ep1, ep2]; % Save the best parameters
                            
                            % Print detailed results of the new "best" found to console and file
                            fprintf(strcat('acc:%.4f±%.4f ','  \n'),avgAcc, stdAcc);
                            fprintf(fid,strcat('acc:%.4f±%.4f ','  \n'),avgAcc, stdAcc);
                            fprintf(strcat('nmi:%.4f±%.4f ','  \n'),avgNmi, stdNmi);
                            fprintf(fid,strcat('nmi:%.4f±%.4f ','  \n'),avgNmi, stdNmi);
                            fprintf(strcat('purity:%.4f±%.4f ','  \n'),avgPurity, stdPurity);
                            fprintf(fid,strcat('purity:%.4f±%.4f ','  \n'),avgPurity, stdPurity);
                            fprintf(strcat('F1:%.4f±%.4f ','  \n'),avgF1, stdF1);
                            fprintf(fid,strcat('F1:%.4f±%.4f ','  \n'),avgF1, stdF1);
                            fprintf(strcat('AR:%.4f±%.4f ','  \n'),avgAR, stdAR);
                            fprintf(fid,strcat('AR:%.4f±%.4f ','  \n'),avgAR, stdAR);
                            fprintf(strcat('Recall:%.4f±%.4f ','  \n'),avgRecall, stdRecall);
                            fprintf(fid,strcat('Recall:%.4f±%.4f ','  \n'),avgRecall, stdRecall);
                            fprintf(strcat('TimePerTrain:%.4f±%.4f ','  \n'),avgTimePerTrain, stdTimePerTrain);
                            fprintf(fid,strcat('TimePerTrain:%.4f±%.4f ','  \n'),avgTimePerTrain, stdTimePerTrain);
    
                            % Store the best metrics and matrices found so far
                            bestAcc = avgAcc;
                            bestStdAcc = stdAcc;
                            bestNmi = avgNmi;
                            bestStdNmi = stdNmi;
                            bestPurity = avgPurity;
                            bestStdPurity = stdPurity;
                            bestF1 = avgF1;
                            bestStdF1 = stdF1;
                            bestAR = avgAR;
                            bestStdAR = stdAR;
                            bestRecall = avgRecall;
                            bestStdRecall = stdRecall;
                            bestTimePerTrain = avgTimePerTrain;
                            bestStdTimePerTrain = stdTimePerTrain;
                            
                            bestC = C;              % Best coefficient matrix
                            bestE = E;              % Best error matrix
                            bestP = P;              % Best projection matrix
                            bestPredict = labelPredict; % Best predicted label set
                            
                            cishu = cishu + 1;      % Increment best-record counter
                            if mod(cishu, 100) == 0 
                                matlab.checkpoint;   % Heuristic check to save state or keep alive (internal function)
                            end
                        end 
                    end
                end
            end
        end
    end
end

% Final output section
fprintf('\n---------------------------------------------------------\n');
fprintf(fid,'\n---------------------------------------------------------\n');

% Print a summary of the absolute best results and the parameters used to achieve them
fprintf(strcat(' acc:%.4f±%.4f \n nmi:%.4f±%.4f \n purity:%.4f±%.4f \n F1:%.4f±%.4f \n AR:%.4f±%.4f \n Recall:%.4f±%.4f \n timeTrain:%.4f±%.4f \n',paraString,'  \n'),bestAcc,bestStdAcc,bestNmi,bestStdNmi,bestPurity,bestStdPurity,bestF1,bestStdF1,bestAR,bestStdAR,bestRecall,bestStdRecall,bestTimePerTrain,bestStdTimePerTrain,paraBest);
fprintf(fid,strcat(' acc:%.4f±%.4f \n nmi:%.4f±%.4f \n purity:%.4f±%.4f \n F1:%.4f±%.4f \n AR:%.4f±%.4f \n Recall:%.4f±%.4f \n timeTrain:%.4f±%.4f \n',paraString,'  \n'),bestAcc,bestStdAcc,bestNmi,bestStdNmi,bestPurity,bestStdPurity,bestF1,bestStdF1,bestAR,bestStdAR,bestRecall,bestStdRecall,bestTimePerTrain,bestStdTimePerTrain,paraBest);

fclose(fid);                % Close the log file

allTime = toc(start0)       % Display the total wall-clock time elapsed for the entire script