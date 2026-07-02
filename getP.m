function P = getP(X, C, E, L, Q, J, beta, gamma, mu)
    % This function computes the updated Projection Matrix P.
    P = (beta*Q+2*gamma*E*L*E'+mu*X*X'-mu*X*C'*X'-mu*X*C*X')\(-X*J'+X*C*J'-mu*X*(-E)'-mu*X*C*C'*X'+mu*X*C*(-E)');
    % The code below is a closed-form solution for P derived from the gradient.
        % --- LEFT SIDE: THE COEFFICIENT MATRIX (Denominator of the derivative) ---
        % beta*Q: Regularization term for P (e.g., Tikhonov or row-sparsity regularization).
        % 2*gamma*E*L*E': Manifold regularization term. 
        % It ensures that the projection is consistent with the graph structure L and error E.
        % mu*X*X' - mu*X*C'*X' - mu*X*C*X': 
        % These terms come from the Augmented Lagrangian penalty relating to the 
        % self-representation error term: ||P'X - P'XC - E||.
        % --- RIGHT SIDE: THE TARGET MATRIX (Numerator of the derivative) ---
        % -X*J' + X*C*J': Gradient components derived from the Lagrangian multiplier J.
        % This term enforces the constraint P'X = P'XC + E via dual ascent.
        % mu*X*(-E)': Penalty term interacting with the sparse error matrix E.
        % mu*X*C*C'*X': Penalty term accounting for the interaction between coefficients C.
        % mu*X*C*(-E)': Interaction term between the coefficients C and the error E.
        
end