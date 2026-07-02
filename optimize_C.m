function C = optimize_C(X, P, F, E, J1, J2, W, R, lambda, gamma, mu, max_iter, tol)
    % Get data dimensions: d = number of features, n = number of samples
    [d, n] = size(X);
    % Get the number of columns in the spectral embedding matrix F
    k = size(F, 2);
    
    % Initialize C as a matrix where each row sums to 1, all elements are equal, 
    % and the diagonal is set to 0 to avoid the trivial self-representation solution.
    C = (ones(n) - eye(n)) / (n-1);
    
    % Pre-compute the projected data A in the reduced space (d x n)
    A = P' * X; 
    % Compute the target matrix for the reconstruction error (incorporating dual variables J1)
    B_matrix = E - J1 / mu;
    % Compute the target matrix for the weighted C term (incorporating dual variables J2)
    D_matrix = R - J2 / mu;
    
    % Compute pairwise Euclidean distances between all projected points in A
    % This represents spatial similarity in the latent space.
    d_matrix = pdist2(A', A', 'euclidean'); 
    
    % Compute pairwise Euclidean distances between rows of the spectral embedding F
    % This helps enforce that points in the same cluster have similar coefficients.
    f_matrix = pdist2(F, F, 'euclidean');
    
    % Main iteration loop to refine matrix C until convergence or max_iter reached
    for iter = 1:max_iter
        C_prev = C;         % Store C from previous iteration to check for convergence
        AC_prev = A * C_prev; % Pre-calculate A * C to reduce redundant matrix multiplications
        
        % Iterate through each sample (row) to optimize its representation coefficients
        for i = 1:n
            % Calculate the squared L2-norm of the i-th sample in the projected space
            a_i = sum(A(:,i).^2);
            
            % Compute the residual matrix M = A - AC - B
            % Then adjust M to isolate the contribution of the i-th coefficient being optimized
            M = A - AC_prev - B_matrix;
            M = M + A(:,i) * C_prev(i, :); 
            
            % Calculate the linear part of the objective function (term 1)
            % This relates to the reconstruction error and the ALM multiplier J2
            term1_part1 = -mu * (A(:,i)' * M);          
            term1_part2 = -mu * W(i,:) .* D_matrix(i,:);
            term1 = term1_part1 + term1_part2;
            
            % Calculate the linear part related to regularization (term 2)
            % Incorporates spatial distances (lambda) and spectral distances (gamma)
            term2 = lambda * d_matrix(i,:) + 2 * gamma * f_matrix(i,:);
            c = term1 + term2; % Combined linear coefficients for the QP solver
            
            % Define the indices for variables to optimize (all except the diagonal C(i,i))
            vars = [1:i-1, i+1:n]; 
            n_vars = length(vars);
            
            % Construct the Hessian matrix (H) for the quadratic term
            % H(j,j) = mu * (||a_i||^2 + W_ij^2)
            H_diag = mu * (a_i + W(i, vars).^2);
            H = diag(H_diag);
            
            % Extract the linear coefficient vector for the current sub-problem
            f_vec = c(vars);
            
            % Set equality constraints: The sum of coefficients in the row must equal 1
            Aeq = ones(1, n_vars);
            beq = 1;
            % Set lower bound: Coefficients must be non-negative (C >= 0)
            lb = zeros(n_vars, 1);
            
            % Configure options for the Quadratic Programming solver
            options = optimoptions('quadprog', 'Display', 'none', 'Algorithm', 'interior-point-convex');
            % Ensure matrices are in double precision and on the CPU (gather from GPU if necessary)
            H = double(gather(H));
            f_vec = double(gather(f_vec));
            
            try
                % Solve the QP problem: min 0.5*y'*H*y + f_vec'*y
                y = quadprog(H, f_vec', [], [], Aeq, beq, lb, [], [], options);
            catch
                % If solving fails (numerical instability), add small regularization to H and retry
                H_reg = H + 1e-8 * eye(n_vars); 
                y = quadprog(H_reg, f_vec', [], [], Aeq, beq, lb, [], [], options);
            end
            
            % Update the i-th row of C with the optimized coefficients
            C(i, vars) = y';
            % Explicitly set the diagonal element to 0
            C(i, i) = 0;
        end
        
        % Calculate the change in matrix C using the Frobenius norm
        diff = norm(C - C_prev, 'fro');
        % Check if the change is below the tolerance to exit early
        if diff < tol
            fprintf('Converged after %d iterations.\n', iter);
            break;
        end
    end
end