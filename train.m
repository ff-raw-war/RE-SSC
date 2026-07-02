function [P, E, W, C] = train(X, alpha, beta, gamma, lambda, ep1, ep2, c, epsilon, maxIter)
    % Initialize vectors to store the history of the optimization process
    lossVec = [];   % To store the objective function value at each iteration
    iterVec = [];   % To store the iteration count
    errVec1 = [];   % To store history of constraint error 1
    errVec2 = [];   % To store history of constraint error 2
    errVec3 = [];   % To store history of constraint error 3

    % Optimization parameters configuration
    rho = 1.01;           % Growth rate of the penalty parameter mu
    mu = 0.1;             % Initial penalty parameter for the augmented Lagrangian terms
    maxMu = 10^5;         % Maximum allowable value for mu
    threshold = 2*10^-4;  % Error threshold for convergence (not explicitly used in current logic)
    
    [d, n] = size(X);     % Get data dimensions: d = features, n = samples
    
    % Variable Initialization: Initializing matrices with random values or zeros
    C = rand(n,n);        % Coefficient matrix (self-representation)
    Lc = getL(C);         % Calculate initial Laplacian matrix based on C
    P = rand(d,d);        % Projection/Transformation matrix
    E = rand(d,n);        % Error/Sparse noise matrix
    R = rand(n,n);        % Auxiliary matrix for regularization on C
    F = rand(n,n);        % Auxiliary matrix for spectral constraints (related to class count c)
    S = rand(d,n);        % Auxiliary matrix for regularization on E
    
    % Lagrangian Multipliers (Dual Variables) initialization
    J1 = zeros(d,n);      % Multiplier for the constraint: P'X - P'XC - E = 0
    J2 = zeros(n,n);      % Multiplier for the constraint: W.*C - R = 0
    J3 = zeros(d,n);      % Multiplier for the constraint: E - S = 0
    
    % Initialize Reweighted Matrix W based on the initial sparse structure of C
    W = ep2./(abs(C)+ep1);
    
    % Initial error settings to enter the loop
    err1 = 10*threshold; 
    errVec1 = [errVec1 err1];
    err2 = 10*threshold;
    errVec2 = [errVec2 err2];
    err3 = 10*threshold;
    errVec3 = [errVec3 err3];

    % Calculate the initial total loss (objective function value)
    loss = getLoss(X, C, P, E, Lc, W, alpha, beta, gamma);
    lossVec = [lossVec ; loss]; 
    iter = 1;
    iterVec = [iterVec iter];
    notConverged = 1;      % Loop control flag

    % Main Iterative Optimization Loop
    while notConverged
        iter = iter + 1;   % Increment iteration counter
        loss0 = loss;      % Record loss from the previous iteration to check for convergence

        % Update QC: Diagonal weight matrix for reweighted row-sparsity of C
        tic;
        QC = diag(0.5 ./ sqrt(sum(C .* C, 2) + eps));
        timeQC = toc       % Benchmark time taken to calculate QC
        
        % Update QP: Diagonal weight matrix for reweighted row-sparsity of P
        tic;
        QP = diag(0.5 ./ sqrt(sum(P .* P, 2) + eps));
        timeQP = toc       % Benchmark time taken to calculate QP
        
        % Update W: Adaptive weights to encourage sparsity in Coefficient matrix C
        W = ep2./(abs(C)+ep1);

        % Update C (Coefficient Matrix): Solving the sub-problem for C
        tic;
        C = optimize_C(X, P, F, E, J1, J2, W, R, gamma, lambda, mu, maxIter, epsilon);
        timeC = toc        % Benchmark time taken to optimize C

        % Update Lc: Recalculate Graph Laplacian for the current C
        Lc = getL(C);

        % Update F and L: Update spectral embedding and Laplacian for clustering constraints
        [F, L] = getF(C, c);
        
        % Update E (Sparse Error Matrix): Minimizing reconstruction error
        tic;
        E = getE(X, P, C, S, Lc, J1, J3, gamma, mu);
        timeE = toc        % Benchmark time taken to optimize E

        % Update R (Regularization auxiliary): Update based on weights and C
        tic;
        R = getR(X, C, W, QC, J2, mu);
        timeR = toc;       % Benchmark time taken to optimize R

        % Update P (Projection Matrix): Solving sub-problem for dimensionality reduction
        tic;
        P = getP(X, C, E, L, QP, J1, beta, gamma, mu);
        timeP = toc        % Benchmark time taken to optimize P
        
        % Update S (Sparsity auxiliary): Update based on error matrix E
        tic;
        S = getS(E, J3, alpha, mu);
        timeS = toc;       % Benchmark time taken to optimize S

        % Update Lagrangian Multipliers (Dual Ascent step)
        % Updates the multipliers based on the residual of the equality constraints
        J1 = J1 + mu*(P'*X-P'*X*C-E); % Enforce reconstruction: P'X = P'XC + E
        J2 = J2 + mu*(W.*C-R);        % Enforce C regularization constraint
        J3 = J3 + mu*(E-S);           % Enforce E regularization constraint

        % Update mu: Increase the penalty strength for the next iteration
        mu = min(rho*mu, maxMu); 
        
        % Calculate current reconstruction error (Frobenius norm)
        err = norm(P'*X-P'*X*C,'fro');

        % Calculate new objective function value
        loss = getLoss(X, C, P, E, Lc, W, alpha, beta, gamma);
        
        % Store errors and loss history
        errVec1 = [errVec1 err1];
        errVec2 = [errVec2 err2];
        errVec3 = [errVec3 err3];
        lossVec = [lossVec loss];
        iterVec = [iterVec iter]; 
        
        % Check Termination Condition 1: Reach maximum iterations
        if iter >= maxIter
            notConverged = 0;
        end
        
        % Check Termination Condition 2: Objective function stabilization
        % Terminate if the change in loss is smaller than epsilon OR if loss starts increasing
        if ((loss0-loss)'*(loss0-loss) < epsilon) || (loss > loss0)
            notConverged = 0;
        end 
        
    end % End of while loop
end % End of function