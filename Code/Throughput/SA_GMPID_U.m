function [E_u,Var_u]=SA_GMPID_U(y,H,Ssigma_n,N_ite,K,M)
%  This program give a scale GMPID which will fix the convergence of the
%  GMPID. For \beta<1, the SA-GMPID can always converge to the MMSE. However, 
%  for the GMPID, it converges to the MMSE detection only when \beta<1/6.
%  \beta = K/M.

%%
Ssigma_x = 1;
Ssig_hat = ( sqrt( (Ssigma_n*Ssigma_x^(-1) + M - K)^2 + 4*(K-1)*Ssigma_n*Ssigma_x^(-1) ) - (Ssigma_n*Ssigma_x^(-1) + M - K) ) / (2*(K-1)*Ssigma_x);
gamma = 1/(K+Ssigma_n/Ssig_hat);

if K<M
%     gamma = (M+Ssigma_n)^-1;
    w = 1 / (1 + gamma*K);   % w = 1 / (1 + \beta );
end

%  no. 2    a upper bound of eigenvalues K<M case
%    A = gamma*(H'*H-diag(diag(H'*H))) + eye(K);
%     w = 2/min(max(sum(abs(A),1)), max(sum(abs(A),2)) );


H2_o = H.^2;

sq_w = sqrt(w);
H = sq_w * H;
y = sq_w * y;

E_y = y;

Var_n = Ssigma_n * ones(M,1);       %variance of noise

H_inv=H.^-1;

Varx = ones(K,1); 
Wx = Varx.^-1;

E_us = zeros(K,M);                  %user node to sum node
Var_us = H2_o'.*(Varx*ones(1,M));   %user node to sum node

 E_u = zeros(K,length(N_ite));
 Var_u = zeros(K,length(N_ite));
 label=1;
 
%%
% case: K<M
if K<M
    for i=1:N_ite(length(N_ite))

       T_su = [E_y - sum(E_us,1)'; Var_n + sum(Var_us,1)'] * ones(1,K) - [ -E_us' ; Var_us'] ;
        E_su = T_su(1:M,1:K);
        Var_su = T_su(M+1:2*M,1:K);

        W_su = Var_su.^(-1).*H2_o;  
        WE_su = W_su.*H_inv.*E_su*w;      %%produce a h'*v^-1*E
        
        if i == N_ite(label)
            Var_u(:,label) = (Wx + sum(W_su,1)').^-1;                           
            E_u(:,label) = Var_u(:,label).*sum(WE_su,1)' - (w-1)*E_us(:,1)./H(1,:)';
            label = label + 1;
        end

        
        T_us = [Wx + sum(W_su,1)'; sum(WE_su,1)'] * ones(1,M);  % modified
        W_us = T_us(1 : K, 1 : M);
        WE_us = T_us(K + 1 : 2 * K, 1 : M);
        Var_us = W_us.^(-1);                                                 
        E_us = Var_us .* WE_us - (w-1)*E_us./H';        %%produce a h'*v^-1*E            

        Var_us = Var_us.*(H2_o');  
        E_us = E_us.*(H');
        
        
    end
end
end

