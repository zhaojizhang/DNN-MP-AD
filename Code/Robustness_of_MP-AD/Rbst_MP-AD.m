%% Program illustration
%  In this program, the message passing algorithm is employedfor  
%  the sparse channel estimation of the Random-Access channel. 
%  The system model is described as follows.
%                           Y = HS + N,     
%  where S is a sparse signal that needs to be recover.
%  H      size: (M, Ns), is the channel matrix. Its entries are i.i.d. with N(0, 1);
%  N      size: (M, Np), the variance of N is \sigma_n^2/Nc;
%  Y      size: (M, Np), the received signal matrix;
%  S      size:  (Ns, Np), user-preamble indicator (0,1) matrix, whose rows denote 
%           size: the users and columns denote the preambles. Each row  
%            of S has only one "1", or be all "0".
%
%                                                                                   ---Lei Liu 
%                                                                                  14/07/2016
%                                                                              Xidian University
%                                                                         leiliuxidian@gmail.com


%% 
clc; clear; close all;
Max = 10^10;                                                          % length of the preamble sequences, which is used to reduce the variance of the noise.
Ns = 150;                                                            % number of users (sources) 
Np = 4;                                                              % number of preambles
M = 40:20:120;                                                              % M=2M*
Pa= 0.1;          %with later approximation Pa=1 IS FORBIDDEN!!!!!!! % active probability of a M2M device; the probability s-th row of S has just one "1"                  
%P0= Pa/Np;                                                  % the probability that the s-th device select the p-th preamble
blocks = 120000;
N_ite = 1:1:10;
Sigma_x = 1; 
Sigma_e = 0.3; % fixed
snr = 0;                                               % -30:10:50;  dB
Sigma_n = Sigma_x ./(10.^(0.1*snr));    % variance of the entries in noise matrix N.
BER = zeros(length(M), length(N_ite));

%% main program
for m = 1 : length(M)
    for kk = 1 : blocks%repeat simulations of H matrixes
        b = randi([1 Max], Ns, 1);
        b(b > (Pa * Max)) = 0;
        b(b >0) = 1;                                                 % active user indicator vector
        kk
        S = zeros(Ns, Np);
        for i = 1 : Ns
            if b(i)==1
                S(i, randi([1 Np]))=1;                        % user-preamble indicator matrix, sparse matrix
            end            
        end
        
        N = sqrt(Sigma_n) * randn(M(m), Np);  % noise producing     
        
        H = randn(M(m), Ns);                                      % channel matrix generate  1/sqrt(K)*randn(M,K); 
        Y = H * S + N;                                              % pass through the fading channel
        He= H + Sigma_e*randn(M(m), Ns);
        
        % Messag passing estimaiton, output LLR, Three dimension (N_ite, Ns, Np)
        L = Message_Passing_Channel_Estimation(Y, He, Np, Ns, M(m), Pa, Sigma_n, N_ite);  
        
       
        L(L>0)=1;                                                    % Decision
        L(L<=0)=0;                                                  % Decision
        F=permute(L(length(N_ite),:,:),[2,3,1]);                    %eighth iteration decision
        
        for i = 1 : length(N_ite)
            BER(m, i) = BER(m, i) + sum(sum( abs( S -  permute(L(i, :, :), [2, 3, 1] ) ) ) ); 
        end
    end
    
end

BER = BER/(blocks*Ns*Np);

save He03MAffectNs150Np4SNR0Pa01;
%%
figure(3);
semilogy(M,BER(:,1)','b-d');
hold on;
legend('SMP');
xlabel('Antenna Number','FontSize',12); %
ylabel('BER','FontName','Times New Roman','FontSize',12);
title('He03MAffectNs150Np4SNR0Pa01');
for i = 2:length(N_ite)
semilogy(M,BER(:,i)','b-d');
saveas(gcf,'He03MAffectNs150Np4SNR0Pa01.fig');
hold on;
end




