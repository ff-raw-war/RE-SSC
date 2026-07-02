function U = getU(X, P, J3, beta, mu)
    % This function solves the sub-problem for matrix U to minimize the Nuclear Norm.
    % The objective is: min beta * ||U||_* + (mu/2) * ||U - (P'X + J3/mu)||_F^2

    % Calculate the shrinkage threshold (epsilon).
    % 'beta' is the weight of the nuclear norm, and 'mu' is the ALM penalty parameter.
    es = beta / mu;

    % Compute the "target" matrix that we want to approximate with a low-rank matrix.
    % This combines the projected data (P'*X) with the dual variable (J3) scaled by mu.
    temp_U = P' * X + J3 / mu;

    % Perform the economy-size Singular Value Decomposition (SVD).
    % uu: Left singular vectors, ss: Singular values (diagonal), vv: Right singular vectors.
    [uu, ss, vv] = svd(temp_U, 'econ');

    % Extract singular values from the diagonal matrix into a column vector.
    ss = diag(ss);

    % Find the number of singular values that are greater than the threshold 'es'.
    % SVP (Singular Value Partitioning) represents the "rank" of the thresholded matrix.
    SVP = length(find(ss > es));

    % Apply the Soft-Thresholding operator to the singular values.
    if SVP > 1
        % Keep significant singular values and shrink them towards zero by 'es'.
        ss = ss(1:SVP) - es;
    else
        % If no singular values (or only one) exceed the threshold, 
        % force the matrix to effectively zero or the lowest possible rank.
        SVP = 1;
        ss = 0;
    end

    % Reconstruct the matrix U using the modified singular values.
    % We multiply the truncated singular vectors by the new diagonal singular values.
    U = uu(:, 1:SVP) * diag(ss) * vv(:, 1:SVP)';
end