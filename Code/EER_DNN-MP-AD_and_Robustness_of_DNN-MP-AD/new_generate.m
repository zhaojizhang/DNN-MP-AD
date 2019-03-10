clc; clear; close all;
Max = 10^10;
Nc = 4;                                                          % length of the preamble sequences, which is used to reduce the variance of the noise.
Ns = 20;                                                            % number of users (sources) 
Np = 4;                                                              % number of preambles
M = 16;                                                              % number of antennas at the base station
Pa= 0.1;          %with later approximation Pa=1 IS FORBIDDEN!!!!!!! % active probability of a M2M device; the probability s-th row of S has just one "1"                  
snr = 10;
P0= Pa/Np;                                                  % the probability that the s-th device select the p-th preamble
blocks = 2000000;
Sigma_x = 1;                                                     % fixed
                                               % -30:10:50;  dB
Sigma_n = Sigma_x ./(10.^(0.1*snr))/Nc;    % variance of the entries in noise matrix N.
lsp=log(P0/(1-P0));
%% main program
SSS=zeros(blocks,Ns*Np);
HHH=zeros(blocks,M*Ns*Np);
YYY=zeros(blocks,M*Np);
    for kk = 1 : blocks %repeat simulations of H matrixes
        b = randi([1 Max], Ns, 1);
        b(b > (Pa * Max)) = 0;
        b(b >0) = 1;                                                 % active user indicator vector
        kk
        temps=[];
        tempy=[];
        temph=[];
        S = zeros(Ns, Np);
        for i = 1 : Ns
            if b(i)==1
                S(i, randi([1 Np]))=1;
                % user-preamble indicator matrix, sparse matrix
            end 
            temps=[temps S(i,:)];
        end       
        
        N = sqrt(Sigma_n) * randn(M, Np);  % noise producing     
        
        H = randn(M, Ns);                  
        
        for h=1:Ns
        temph=[temph kron(ones(1,Np),H(:,h)')];
        end
               
        Y = H * S + N;
        for j=1:Np
            tempy=[tempy Y(:,j)'];
        end     
        SSS(kk,:)=temps;
        HHH(kk,:)=temph;
        YYY(kk,:)=tempy;            

    end
        csvwrite('trainY.csv',YYY);
        csvwrite('trainH.csv',HHH);
        csvwrite('trainS.csv',SSS);
blocks = 500000;
SS=zeros(blocks,Ns*Np);
HH=zeros(blocks,M*Ns*Np);
YY=zeros(blocks,M*Np);
    for kk = 1 : blocks %repeat simulations of H matrixes
        b = randi([1 Max], Ns, 1);
        b(b > (Pa * Max)) = 0;
        b(b >0) = 1;                                                 % active user indicator vector
        kk
        temps=[];
        tempy=[];
        temph=[];
        S = zeros(Ns, Np);
        for i = 1 : Ns
            if b(i)==1
                S(i, randi([1 Np]))=1;
                % user-preamble indicator matrix, sparse matrix
            end 
            temps=[temps S(i,:)];
        end       
        
        N = sqrt(Sigma_n) * randn(M, Np);  % noise producing     
        
        H = randn(M, Ns);                  
        
        for h=1:Ns
        temph=[temph kron(ones(1,Np),H(:,h)')];
        end
               
        Y = H * S + N;
        for j=1:Np
            tempy=[tempy Y(:,j)'];
        end     
        SS(kk,:)=temps;
        HH(kk,:)=temph;
        YY(kk,:)=tempy;            

    end
        csvwrite('testY.csv',YY);
        csvwrite('testH.csv',HH);
        csvwrite('testS.csv',SS);






   