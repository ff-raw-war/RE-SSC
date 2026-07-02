function S = getS(E, J3, alpha, mu)
    % This function updates the auxiliary matrix S using the Soft-Thresholding operator.
    % In optimization, this is the proximal operator for the L1-norm regularization term (alpha * ||S||_1).
    
    % Step-by-step breakdown:
    % 1. (E + J3/mu): This combines the current error matrix 'E' with the Lagrangian multiplier 
    %    (dual variable) 'J3', scaled by the penalty parameter 'mu'. This represents the 
    %    "target" values that S wants to match.
    % 
    % 2. (alpha / mu): This is the shrinkage threshold. 
    %    - alpha: The hyperparameter controlling the importance of sparsity.
    %    - mu: The current ALM penalty parameter. 
    %    As mu increases over iterations, the threshold becomes smaller, allowing more 
    %    precise reconstruction.
    %
    % 3. soft_threshold1: This function applies the formula: sign(x) * max(|x| - threshold, 0)
    %    It pushes small values (noise) to exactly zero and shrinks other values toward zero.
    S = soft_threshold1(E+J3/mu, alpha/mu);
end