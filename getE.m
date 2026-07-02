function E = getE(X, P, C, S, L, J1, J3, gamma, mu)
    % This function calculates the optimal matrix E by solving the matrix equation 
    % derived from the first-order optimality condition of the objective function.

    % Pre-calculate the product of the projection matrix P and its transpose.
    % This is done to improve computational efficiency as it is used multiple times below.
    PP = P*P';

    % Define the coefficient matrix 'A' for the Lyapunov equation.
    % This term typically corresponds to the quadratic penalty from the 
    A = 2*mu*(P*P');

    % Define the coefficient matrix 'B' for the Lyapunov equation.
    % This term represents the Graph Laplacian regularization (smoothing) 
    % applied to the error matrix E, scaled by the hyperparameter 'gamma'.
    B = 2*gamma*L;

    % Calculate the constant (inhomogeneous) matrix 'C' for the Lyapunov equation.
    % It aggregates the effects of:
    % 1. Dual variables (Lagrangian multipliers) J3 and J1.
    % 2. The projection of the data (P'*X) and its reconstruction (P'*X*C).
    % 3. The auxiliary sparsity variable S.
    % The backslash (\) operator performs pre-multiplication by the inverse of (P*P').
    C = PP\J3 - PP\J1 - mu*PP\P'*X + mu*PP\P'*X*C - mu*PP\S;

    % 'gather' is used to transfer data from the GPU memory to the CPU memory.
    % This is necessary because MATLAB's 'lyap' function is a standard numerical 
    % routine that typically executes on the CPU, not the GPU.
    A = gather(A);
    B = gather(B);
    C = gather(C);
    
    % Solve the Continuous Lyapunov Equation: A*E + E*B + C = 0.
    % Here, 'lyap' computes the unique matrix E that satisfies the optimality 
    % condition for the E-subproblem in this iteration.
    E = lyap(A, B, C);
    
end