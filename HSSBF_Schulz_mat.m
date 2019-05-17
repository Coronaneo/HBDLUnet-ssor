function X = HSSBF_Schulz_mat(A,X,maxIter,order)
% Input:
% A: the matrix to be inverted
% X: an approximate inverse of A
% maxIter: the maximumn iteration number
%
% Output:
% X: a better approximate inverse of A using the Schultz iteration

if nargin < 2, maxIter = 5; end
switch order
    case 2
        for cnt = 1:maxIter
            X = X*(2*eye(size(X))-A*X);
        end
    case 3
        for cnt = 1:maxIter
            X = X*(3*eye(size(X))-3*A*X+A*A*X*X);
        end
end
end