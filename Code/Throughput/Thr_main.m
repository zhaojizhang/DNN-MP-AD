%% 
clc; clear; close all;
Max = 10^10;                                                          % length of the preamble sequences, which is used to reduce the variance of the noise.
Ns = 300;                                                            % number of users (sources) 
Np = 6;                                                              % number of preambles
M = 140;                                                              % M=2M*  number of antennas at the base station
Pa= 0.06:0.04:0.3;          %with later approximation Pa=1 IS FORBIDDEN!!!!!!! % active probability of a M2M device; the probability s-th row of S has just one "1"                  
%P0= Pa/Np;                                                  % the probability that the s-th device select the p-th preamble
blocks = 1e5;
N_ite = 1:1:10;
S_ite=1:1:30;
Sigma_x = 1;                                                     % fixed
SNR = 25;                                               % -30:10:50;  dB
Sigma_n = Sigma_x./(10.^(0.1*SNR));    % variance of the entries in noise matrix N.
TRP_SMP = zeros(length(Pa), 1);
TRP_LMMSE = zeros(length(Pa), 1);
Fail_Rec_SA=zeros(length(Pa), 1);
Fail_Rec_LMMSE=zeros(length(Pa), 1);
Fail_SMP=zeros(length(Pa), 1);
Rec_th=2e-4;

%% main program
parfor m=1:length(Pa)
    Pa(m)
for kk = 1 : blocks%repeat simulations of H matrixes
    kk
        b = randi([1 Max], Ns, 1);
        b(b > (Pa(m) * Max)) = 0;
        b(b >0) = 1;% active user indicator vector
        
        S = zeros(Ns, Np);
        for i = 1 : Ns
            if b(i)==1
                S(i, randi([1 Np]))=1;                        % user-preamble indicator matrix, sparse matrix
            end            
        end
        
        N = sqrt(Sigma_n) * randn(M, Np);  % noise producing     
        
        H = randn(M, Ns);                                      % channel matrix generate  1/sqrt(K)*randn(M,K); 
        Y = H * S + N;                                              % pass through the fading channel
        % Messag passing estimaiton, output LLR, Three dimension (N_ite, Ns, Np)
        L = Message_Passing_Channel_Estimation(Y, H, Np, Ns, M, Pa(m), Sigma_n, N_ite);  
        Dec_SMP=permute( L(length(N_ite),:,:), [2, 3, 1]);
        Dec_SMP(Dec_SMP>0)=1;
        Dec_SMP(Dec_SMP<=0)=0;
        temp_sa=zeros(Np,1);
        temp_lmmse=zeros(Np,1);
        Temp_Fail_Rec_SA=zeros(Np,1);
        Temp_Fail_Rec_LMMSE=zeros(Np,1);
        Temp_Fail_SMP=zeros(Np,1);
        for p=1:Np
            Tot=sum(S(:,p));
            if (sum( abs( Dec_SMP(:,p)-S(:,p) ) )<0.5) && (Tot>0)   % Correct activity detection
               x = randn(2*Tot,1); % Both the Re part and the Im part
               n = sqrt(Sigma_n) * randn(M,1);
               Ha=H(:,S(:,p)>0);
               HA=[Ha(1:M/2,:) -Ha(M/2+1:M,:);
                   Ha(M/2+1:M,:) Ha(1:M/2,:)];
               y=HA*x+n;
               if M>length(x)
               [x_sa,~] = SA_GMPID_U(y,HA,Sigma_n,S_ite,length(x),M);
               else
               [x_sa,~] = SA_GMPID_O(y,HA,Sigma_n,S_ite,length(x),M);
               end
               x_lmmse =  (HA'*HA+Sigma_n*eye(2*Tot))\(HA'*y);
               err_sa=(x_sa(1:Tot,length(S_ite))-x(1:Tot))+1i*(x_sa(Tot+1:2*Tot,length(S_ite))-x(Tot+1:2*Tot));
               err_lmmse=(x_lmmse(1:Tot)-x(1:Tot))+1i*(x_lmmse(Tot+1:2*Tot)-x(Tot+1:2*Tot));
               for t=1:Tot
                   if abs(err_sa(t))^2<Rec_th
                     temp_sa(p)=temp_sa(p)+1;
                   else
                     Temp_Fail_Rec_SA(p)=Temp_Fail_Rec_SA(p)+1;  
                   end
                   if abs(err_lmmse(t))^2<Rec_th
                     temp_lmmse(p)=temp_lmmse(p)+1;
                   else
                     Temp_Fail_Rec_LMMSE(p)=Temp_Fail_Rec_LMMSE(p)+1;
                   end
               end
            else
                Temp_Fail_SMP(p)=Tot;
            end
        end
        TRP_SMP(m)=TRP_SMP(m)+sum(temp_sa);
        TRP_LMMSE(m)=TRP_LMMSE(m)+sum(temp_lmmse);
        Fail_Rec_SA(m)=Fail_Rec_SA(m)+sum(Temp_Fail_Rec_SA);
        Fail_Rec_LMMSE(m)=Fail_Rec_LMMSE(m)+sum(Temp_Fail_Rec_LMMSE);
        Fail_SMP(m)=Fail_SMP(m)+sum(Temp_Fail_SMP);
end
end
TRP_SMP=TRP_SMP/blocks;
TRP_LMMSE=TRP_LMMSE/blocks;
Fail_Rec_SA=Fail_Rec_SA/blocks;
Fail_Rec_LMMSE=Fail_Rec_LMMSE/blocks;
Fail_SMP=Fail_SMP/blocks;
save M140_2e-4th_SNR25_300User




