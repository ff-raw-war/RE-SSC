function [F, L] = getF(C, c)
    % This function computes the Graph Laplacian and the spectral embedding matrix F.
    % Inputs: 
    %   C: The sparse coefficient/similarity matrix (n x n).
    %   c: The target number of clusters (classes).
    % Outputs:
    %   F: The spectral embedding matrix (n x c), containing c eigenvectors.
    %   L: The computed Graph Laplacian matrix.
    % Step 1: Symmetrize the coefficient matrix.
    % Subspace clustering often produces a directed (unsymmetric) matrix C.
    % Spectral clustering requires a symmetric affinity matrix. 
    % This calculates the average of C and its transpose to ensure LC(i,j) == LC(j,i).
    LC = (C+C')/2;
    % Step 2: Construct the unnormalized Graph Laplacian matrix L.
    % L is defined as D - W, where D is the Degree Matrix and W is the Adjacency Matrix.
    % 1. sum(LC) calculates the sum of each column (the 'degree' of each node).
    % 2. diag(...) places these degrees on the diagonal of a matrix D.
    % 3. Subtracting LC (the weights) results in the Laplacian matrix L.
    L = diag(sum(LC)) - LC;
     % Step 3: Perform Eigen-decomposition to find the spectral embedding.
    % This calls 'eig1', a common utility function in subspace clustering research.
    % It calculates the 'c' eigenvectors corresponding to the 'c' smallest eigenvalues.
    % The '0' parameter typically tells the solver to sort eigenvalues in ascending order.
    % F: The resulting matrix where each row is the new low-dimensional coordinate for a sample.
    % ev: Contains the eigenvalues (the 'eigen-spectrum').
    [F, ~, ev] = eig1(L, c, 0);
end