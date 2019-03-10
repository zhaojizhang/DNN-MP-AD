function L = Message_Passing_Channel_Estimation(Y, H, Np, Ns, M, Pa, Sigma_n, N_ite)
%%
% Max = 10^10;
L_Max = 70;
% min = 10^-5;
P0 = Pa/Np;                                                      % the probability that the s-th device select the p-th preamble
L_pa = log( Pa );
Lsp = -log( P0^(-1) - 1 ) * ones(Ns, Np);
Ls = zeros(M, Ns, Np);
Lvs = zeros(M, Ns, Np);

Us = zeros(M, Ns, Np); 
Vs = zeros(M, Ns, Np);

H2 = H.^2;

L = zeros( length(N_ite), Ns, Np);
label=1;

%%
for i = 1 : N_ite(length(N_ite))
   %%  Message update at sum node: s->v

    Pvs = 1 ./( 1 + exp(-Lvs) );                         % take care!!!!
    for p = 1 : Np
        Tem1 = H .* Pvs(:, :, p);
        Us(:, :, p) = sum( Tem1, 2 ) * ones(1, Ns) - Tem1;
        Tem2 =  H2 ./( 2 + exp( Lvs(:, :, p) ) + exp( - Lvs(:, :, p)) ) ;
        Vs(:, :, p) = sum( Tem2, 2) * ones(1, Ns) - Tem2 + Sigma_n;    % Sigma_n * ones(M, Ns)
        
        Ls(:, :, p) = (  (  Y(:, p) * ones(1, Ns) - Us(:, :, p) ) .* H - 0.5 * H2 ) ./  Vs(:, :, p); 
%         Ls(:, :, p) = (  (  Y(:, p) * ones(1, Ns) ) .* H - 0.5 * H2 ) ./  Vs(:, :, p); 
    end
    

   
%%  Message update at variable node: v->c
    Lvc = permute( sum( Ls, 1 ), [2, 3, 1] ) + Lsp;

%%  Message update at check node: c->v
    Tem3 =  log( 1 + exp(Lvc) );    % take care!!!!
    Tem3( Lvc>L_Max )=Lvc( Lvc>L_Max );

    Tem3 =  L_pa - sum(Tem3, 2) * ones(1, Np) + Tem3;
    Lc = -log( exp(-Tem3) - 1 );       % take care!!!!
    Lc(Tem3<-L_Max)=Tem3(Tem3<-L_Max);


%%  Message update at variable node: v->s 
      for p = 1 : Np
            Lvs( :, :, p) = ones(M,1) * ( sum( Ls( :, :, p ), 1 ) + Lsp(: , p)' +  Lc(: , p)' ) - Ls( :, :, p );
      end
    
      if i == N_ite(label)                                       % out put        
        Lv = permute( sum( Ls, 1 ) , [2, 3, 1] ) + Lsp +  Lc  ;
        L( label, :, :) = Lv;                                       % P
        
        label = label + 1;
     end

end


end