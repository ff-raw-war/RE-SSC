function loss = getLoss(X, C, P, E, Lx, W, alpha, beta, gamma)

    l21fsP = 0; % L21norm
    dimP = size(P , 2);
    for i = 1 : dimP
        l21fsP = l21fsP + norm(P( : , i), 2); 
    end
    
    WC = W.*C;
    l21fsWC = 0;
    dimWC = size(WC, 2);
    for i = 1 : dimWC
        l21fsWC = l21fsWC + norm(WC( : , i), 2); 
    end

    loss = l21fsWC +alpha*norm(E,1)+ beta*l21fsP + gamma*trace(P'*X*Lx*X'*P);
    
end