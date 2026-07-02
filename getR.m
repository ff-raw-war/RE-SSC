function R = getR(X, C, W, Q, J2, mu)
    % This function solves the sub-problem for the auxiliary matrix R.
    % It finds the optimal R that minimizes the terms in the objective function
    % involving the constraint: R = W .* C (where .* is element-wise multiplication).

    % Extract the dimensions of the data matrix X.
    % d is the number of features (rows), and n is the number of samples (columns).
    [d, n] = size(X);
        % Breakdown of the calculation:
    % 1. (Q - mu*eye(n)): This is the coefficient matrix.
    %    - Q: A diagonal weight matrix (calculated in the main loop) that typically 
    %      handles row-sparsity or reweighted l2-norm regularization.
    %    - mu*eye(n): The quadratic penalty term from the ALM framework.
    %
    % 2. (J2 + mu*W.*C): This is the target matrix.
    %    - J2: The Lagrangian multiplier (dual variable) associated with the constraint R = W.*C.
    %    - mu*W.*C: The penalty term enforcing the relationship between the weight matrix W,
    %      the coefficient matrix C, and the auxiliary variable R.
    %
    % 3. The backslash (\) operator:
    %    - Efficiently solves the linear equation (A \ B) to find R such that (Q - mu*I) * R = (J2 + mu*W.*C).
    R = (Q - mu*eye(n))\(J2 + mu*W.*C);
end