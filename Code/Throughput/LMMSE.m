clc; clear; close all;
Max = 10^10;                                                          % length of the preamble sequences, which is used to reduce the variance of the noise.
Ns = 200;                                                            % number of users (sources) 
Np = 4;                                                              % number of preambles
M = 180;                                                              % number of antennas at the base station
Pa= 0.1:0.04:0.3;          %with later approximation Pa=1 IS FORBIDDEN!!!!!!! % active probability of a M2M device; the probability s-th row of S has just one "1"                  
%P0= Pa/Np;                                                  % the probability that the s-th device select the p-th preamble
blocks = 3000;
Sigma_x = 1;                                                     % fixed
SNR = 25;                                               % -30:10:50;  dB
Sigma_n = Sigma_x./(10.^(0.1*SNR));    % variance of the entries in noise matrix N.
TRP_LMMSE = zeros(length(Pa), 1);

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
        temp_lmmse=zeros(Np,1);
        for p=1:Np
            Tot=sum(S(:,p));
            if Tot>0   % Correct activity detection
               x = randn(2*Tot,1); % Both the Re part and the Im part
               n = sqrt(Sigma_n) * randn(M,1);
               Ha=H(:,S(:,p)>0);
               HA=[Ha(1:M/2,:) -Ha(M/2+1:M,:);
                   Ha(M/2+1:M,:) Ha(1:M/2,:)];
               y=HA*x+n;
               x_lmmse =  (HA'*HA+Sigma_n*eye(2*Tot))\(HA'*y);
               err_lmmse=(x_lmmse(1:Tot)-x(1:Tot))+1i*(x_lmmse(Tot+1:2*Tot)-x(Tot+1:2*Tot));
               temp_lmmse(p)=sum(abs(err_lmmse).^2);
            end
        end
        TRP_LMMSE(m)=TRP_LMMSE(m)+sum(temp_lmmse)/sum(b);
end
end
TRP_LMMSE=TRP_LMMSE/blocks;





