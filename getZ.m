function Z = getZ(C, J2, mu2)
    % This function solves the sub-problem for the sparse coefficient matrix Z.
    % It minimizes a combination of an L1-norm regularization term
    [n, ~] = size(C); 
    % Get the number of samples 'n' from the size of matrix C.
    % C is the representation matrix from the other sub-problem.
    % The tilde (~) is used to ignore the second dimension (columns).
    Z = soft_threshold1(C + J2/mu2, eye(n)/mu2);
    % Apply the Soft-Thresholding (Shrinkage) operator.
    % This is the proximal operator for the L1-norm, which induces sparsity in Z.
    % 1. (C + J2/mu2): This is the 'target' matrix, combining the current 
    %    estimate C with the scaled dual variable (Lagrange multiplier) J2.
    % 2. (eye(n)/mu2): This sets the threshold for the shrinkage. 
    %    Note: Since eye(n) is used, if the 'soft_threshold1' function handles 
    %    matrix-form thresholds, it primarily applies a 1/mu2 threshold to the 
    %    diagonal and potentially influences how off-diagonal elements are shrunk.
    Z = Z-diag(diag(Z)); 
    % Enforce the 'diag(Z) = 0' constraint.
    % In subspace clustering, this is crucial to prevent the trivial solution 
    % where a point represents itself (e.g., x_i = 1 * x_i).
    % 1. diag(Z) extracts the diagonal elements as a vector.
    % 2. diag(diag(Z)) creates a diagonal matrix from that vector.
    % 3. Subtracting it from Z effectively sets all diagonal entries of Z to exactly zero.
end