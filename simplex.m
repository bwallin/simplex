% [x, zstar, opt_basis] = simplex(A, b, c, starting_basis)
% 
% Function using simplex method to solve linear programming problems of 
% the form:
%     
% maximize cx
%          Ax=b
%           x>=0
% 
% Arguments: 
%     A ~ m x n 
%     b ~ n x 1 
%     c ~ 1 x m
%     starting_basis ~ 1 x m
% 
% Returns:
%     x ~ m x 1 - optimal solution if found ([] otherwise)
%     zstar - optimal value if found ([] otherwise)
%     opt_basis - optimal basis if found ([] otherwise)
%
% Author: Bruce Wallin
% Revised: 4/2/2011

function [x, zstar, opt_basis] = simplex(A, b, c, starting_basis)
% Initialize outputs
x = [];
zstar = [];
opt_basis = [];

[m,n] = size(A);

% Check sizes of inputs for consistency
if any(size(b) ~= [m,1]) || ...
       any(size(c) ~= [1,n]) || ...
       any(size(starting_basis) ~= [1,m])
    err = MException('MATLAB:BadInput', ...
        'Inputs are inconsistant, check that A~[m,n], b~[m,1], c~[1,n], starting_basis~[1,m]');
    throw(err)
end

basis = starting_basis;
nonbasis = setdiff(1:n, starting_basis);

% Check for feasability of basis
B = A(:, basis);
if any(inv(B)*b < 0)
    x = [];
    zstar = [];
    opt_basis = [];
    return
end

% Begin simplex method
while true
    An = A(:, nonbasis);
    B = A(:, basis);
    cn = c(:, nonbasis);
    cb = c(:, basis);
    x = zeros([n,1]);
    x(basis) = inv(B)*b;
    
    y = cb/B;
    rn = cn - y*An;

    % Check for optimality of solution
    if all(rn<=0)
        zstar = c*x;
        opt_basis = basis;
        return
    end

    % Candidates to enter have maxumim rn
    candidates = nonbasis(rn==max(rn));
    % Choose first candidate (minimum subscript rule?)
    enterer = candidates(1);
    a = A(:,enterer);
    d = B\a;

    % Check for unboundedness
    if all(d <= 0)
        x = zeros(n,1);
        zstar = Inf;
        opt_basis = [];
        return
    end

    % Candidates to leave have positive d
    candidates = basis(d > 0);
    % Compute max t for candidates only
    d = d(d > 0);
    t = (diag(d, 0))\x(candidates);
    % Final candidates
    candidates = candidates(t == min(t));
    % Choose first candidate (minimum subscript rule?)
    leaver = candidates(1);
    
    basis = setdiff(basis, leaver);
    basis = union(basis, enterer);
    nonbasis = setdiff(1:n, basis);
end

end    