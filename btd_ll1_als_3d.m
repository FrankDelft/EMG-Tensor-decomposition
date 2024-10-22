function [A, B, C, const, output] = btd_ll1_als_3d(T, R, Lr, options, varargin)
% BTD_LL1_ALS_3D Computes the (Lr,Lr,1) block-term decomposition (BTD) of a
% 3rd-order tensor using the alternative least squares algorithm (ALS).
%
% INPUT:
%   T (I_1 x I_2 x I_3): 3-D tensor data tensor
%   R (1 x 1): number of BTD components (sources)
%   Lr (1 x R): Ranks of the factor matrices in the first and second mode for
%               all components
%   options (struct): optimization options including:
%          - th_relerr (1 x 1): relative error threshold
%          - maxiter   (1 x 1): max number of iterations
%   variable extra inputs (1 x 3 cell array OR 3 arrays, OPTIONAL): 
%           initialization for the factor matrices
%
% OUTPUT:
%   A (I_1 x sum(Lr)): mode-1 factor matrix
%   B (I_2 x sum(Lr)): mode-2 factor matrix 
%   C (I_3 x R): mode-3 factor matrix with normalized columns 
%   const (1 x R):  vector containing the respective weights for the BTD
%           components
%   output (struct) : optimization options including:
%          - numiter (1 x 1): the number of iterations the algorithm ran for
%          - relerr (1 x numiter): relative error achieved, defined as 
%           Frobenius norm of the residual of the decomposition OVER 
%           Frobenius norm of the original tensor.
% 

maxiter=options.maxiter;
th_relerr=options.th_relerr;

% Check if the initialization for the factor matrices was given
init = [];
if ~isempty(varargin)
    if length(varargin) == 1    % Given as cell
        init = varargin{:}; 
    else                        % Given as matrices 
        init = varargin;
    end
end

% Initialize the three factor matrices 
if isempty(init)    
    A = randn(size(T, 1), R*Lr);
    B = randn(size(T, 2), R*Lr);
    C = randn(size(T, 3), R);
else              
    A = init{1};
    B = init{2};
    C = init{3};
end

%normalise columns
A = normc(A);
B = normc(B);
C = normc(C);

% Obtain the three tensor unfoldings 
T1 = hidden_mode_n_matricization(T,1);
T2 = hidden_mode_n_matricization(T,2);
T3 = hidden_mode_n_matricization(T,3);

% ALS iterations
for idxiter = 1:maxiter
    kron_cat1=[];
    for i=1:R
        Br=B(:,1+(i-1)*Lr:i*Lr);
        temp=kron(C(:,i),Br);
        kron_cat1=[kron_cat1 temp];        
    end
    A = T1 * pinv(kron_cat1.');
    % A = normc(A);
    % A = A ./ sqrt(sum(A.^2, 1)); 

    % A = A ./ vecnorm(A); 

    kron_cat2=[];
    for i=1:R
        Ar=A(:,1+(i-1)*Lr:i*Lr);
        temp=kron(C(:,i),Ar);
        kron_cat2=[kron_cat2 temp];        
    end
    B = T2 * pinv(kron_cat2.');
    % B = B ./ sqrt(sum(B.^2, 1));
    % B=normc(B);
    % B = B ./ vecnorm(B); 

    khatri_cat=[];
    for i=1:R
        Ar=A(:,1+(i-1)*Lr:i*Lr);
        Br=B(:,1+(i-1)*Lr:i*Lr);
        temp=hidden_khatri_rao(Br,Ar)*ones([Lr,1]);
        khatri_cat=[khatri_cat temp];        
    end
    C=T3*pinv(khatri_cat.');

    const=vecnorm(C);
    C = C ./ const;

    


    T3_est=C*diag(const)*khatri_cat.';
    
    relerr(idxiter) = norm((T3 - T3_est),"fro") / norm(T3,"fro");

    if relerr(idxiter) < th_relerr 
        break;
    end


end
output.numiter = idxiter;
output.relerr = relerr;

end