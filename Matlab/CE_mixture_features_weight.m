% Algorithm 2 impletation
%
% Input:
%
% "counters_trans" data matrix where the columns 1:p are the feature vector X,
% last column of counters_trans represents the user count set 
%
% These features are repeated 8 times to capture each of the user counts of
% Table 3 given a vector. 
% For simplicity, we assume a countset of size 8 (as opposed to the 6 counts
% of Table 3) to filled the unobserved targeting indicator during the Gibbs
% sampling process.
%
% Given X_i (i=0:feat_Combinations-1) and p vector size, we assume:
% counters_trans(i*8+1,p+1) = N_{{0,1}C}^0
% counters_trans(i*8+2,p+1) = N_{{0,1}C}^1
% counters_trans(i*8+3,p+1) = 0
% counters_trans(i*8+4,p+1) = 0
% counters_trans(i*8+5,p+1) = N_{0S}^0
% counters_trans(i*8+6,p+1) = N_{0S}^1
% counters_trans(i*8+7,p+1) = N_{1S}^0
% counters_trans(i*8+8,p+1) = N_{1S}^1
%
%
% Output:
% 
% regression vector samples Eq 14 (beta_samples, gamma_d0_samples, gamma_n_samples)
%
% Variable mapping
%
% gamma_1C -> gamma_d0
% gamma_1S -> gamma_d1
% gamm_0 -> gamma_n
% beta_sel -> beta
%
% Alternatively, you can uncomment "str" assignment and the "save" lines
% to save intermediate fitting steps

function [beta_samples, gamma_d0_samples, gamma_d1_samples, gamma_n_samples]=...
    CE_mixture_features_weight(counters_trans,N_burnin,N_samples,alpha_0)

% alpha_0 = 0.5;
% N_burnin = 500;
% N_samples = 3000;

% Feature vector size is equal to the number of columns minus 1
% Last column of counters_trans represents the user count set
p = size(counters_trans,2) -  1; 

% Data count indices given a feature vector value X_i
indx_n0_y0 = 1;
indx_n0_y1 = 2;
indx_d0_y0 = 3;
indx_d0_y1 = 4;
indx_n1_y0 = 5;
indx_n1_y1 = 6;
indx_d1_y0 = 7;
indx_d1_y1 = 8;

% Length of count set per feature value
len_count = 8;

% str = 'model_intermediate';

%------------------------
% Initialization of class label for control
counters_samp = counters_trans(:,end) + alpha_0;
theta_n = counters_samp(indx_n1_y1:len_count:end)./...
    (counters_samp(indx_n1_y0:len_count:end) + counters_samp(indx_n1_y1:len_count:end));
theta_d1 = counters_samp(indx_d1_y1:len_count:end)./...
    (counters_samp(indx_d1_y0:len_count:end) + counters_samp(indx_d1_y1:len_count:end));
omega = (counters_samp(indx_d1_y1:len_count:end)+counters_samp(indx_d1_y0:len_count:end))./...
    (counters_samp(indx_d1_y0:len_count:end) + counters_samp(indx_d1_y1:len_count:end)...
    +counters_samp(indx_n1_y0:len_count:end) + counters_samp(indx_n1_y1:len_count:end));
p0_camp = counters_samp(indx_n0_y1:len_count:end)./...
    (counters_samp(indx_n0_y0:len_count:end) + counters_samp(indx_n0_y1:len_count:end));
theta_d0 = (p0_camp - theta_n.*(1-omega))./omega;
theta_d0(theta_d0<0) = p0_camp(theta_d0<0);

Pd_d0_Act = omega.*theta_d0./(omega.*theta_d0+(1-omega).*theta_n);
Pd_d0_nAct = omega.*(1-theta_d0)./(omega.*(1-theta_d0)+(1-omega).*(1-theta_n));

N_d0_Act = binornd(counters_trans(indx_n0_y1:len_count:end,end),Pd_d0_Act);
N_d0_nAct = binornd(counters_trans(indx_n0_y0:len_count:end,end),Pd_d0_nAct);
N_n0_Act = counters_trans(indx_n0_y1:len_count:end,end) - N_d0_Act;
N_n0_nAct = counters_trans(indx_n0_y0:len_count:end,end) - N_d0_nAct;

counters_samp = counters_trans(:,end);
counters_samp(indx_n0_y0:len_count:end) = N_n0_nAct;
counters_samp(indx_n0_y1:len_count:end) = N_n0_Act;
counters_samp(indx_d0_y0:len_count:end) = N_d0_nAct;
counters_samp(indx_d0_y1:len_count:end) = N_d0_Act;


fprintf('Done initializing parameters\n');

%----------Burnin--------------------------------------------
% Same X repeated twice for y0 and y1
X = [counters_trans(indx_d1_y0:len_count:end,1:p);counters_trans(indx_d1_y1:len_count:end,1:p)];
X = [ones(size(X,1),1),X];
nX = size(X,1)/2;
% This base y is in the order of [y0;y1]
y = [counters_trans(indx_d1_y0:len_count:end,end-1)-1;counters_trans(indx_d1_y1:len_count:end,end-1)];
wg_d1 = [counters_samp(indx_d1_y0:len_count:end);counters_samp(indx_d1_y1:len_count:end)];
wg_d1 = wg_d1 + alpha_0;

gamma_d1_samples = sampleProbReg_Approx(X(1:nX,:),wg_d1(nX+1:end),wg_d1(1:nX)+wg_d1(nX+1:end),N_burnin); % -- Fixed during the iterations
fprintf('Done sampling gamma_d1 burnin\n');
%------------------------

block=1;
prop =0.1;
k=1;

for s=1:N_burnin
    if s>=round(prop*k*N_burnin)
        k = k+1;
        fprintf('Drawing sample %d of %d\n',s,N_burnin);
    end
    
    wg_d = [counters_samp(indx_n1_y0:len_count:end)+counters_samp(indx_n1_y1:len_count:end)+...
        counters_samp(indx_n0_y0:len_count:end)+counters_samp(indx_n0_y1:len_count:end);
        counters_samp(indx_d1_y0:len_count:end)+counters_samp(indx_d1_y1:len_count:end)+...
        counters_samp(indx_d0_y0:len_count:end)+counters_samp(indx_d0_y1:len_count:end)];
    wg_d = wg_d + alpha_0;
    beta_s = sampleProbReg_Approx(X(1:nX,:),wg_d(nX+1:end),wg_d(1:nX)+wg_d(nX+1:end),block);
    
    gamma_d1 = gamma_d1_samples(:,s);

    wg_d0 = [counters_samp(indx_d0_y0:len_count:end);counters_samp(indx_d0_y1:len_count:end)];
    wg_d0 = wg_d0 + alpha_0;
    gamma_d0 = sampleProbReg_Approx(X(1:nX,:),wg_d0(nX+1:end),wg_d0(1:nX)+wg_d0(nX+1:end),block);

    wg_n = [counters_samp(indx_n0_y0:len_count:end)+counters_samp(indx_n1_y0:len_count:end);...
        counters_samp(indx_n0_y1:len_count:end)+counters_samp(indx_n1_y1:len_count:end)];
    wg_n = wg_n + alpha_0;
    gamma_n = sampleProbReg_Approx(X(1:nX,:),wg_n(nX+1:end),wg_n(1:nX)+wg_n(nX+1:end),block);
    
    % X partition by half because X is doubled, one half for y=0 and the
    % other for y=1 in the gamma's sampling
    omega = normcdf(X(1:size(X,1)/2,:)*beta_s); % old omega
    theta_d0 = normcdf(X(1:size(X,1)/2,:)*gamma_d0); % old theta_d0
    theta_n = normcdf(X(1:size(X,1)/2,:)*gamma_n); %old theta_n

    Pd_d0_Act = omega.*theta_d0./(omega.*theta_d0+(1-omega).*theta_n);
    Pd_d0_nAct = omega.*(1-theta_d0)./(omega.*(1-theta_d0)+(1-omega).*(1-theta_n));

    N_d0_Act = binornd(counters_trans(indx_n0_y1:len_count:end,end),Pd_d0_Act);
    N_d0_nAct = binornd(counters_trans(indx_n0_y0:len_count:end,end),Pd_d0_nAct);
    N_n0_Act = counters_trans(indx_n0_y1:len_count:end,end) - N_d0_Act;
    N_n0_nAct = counters_trans(indx_n0_y0:len_count:end,end) - N_d0_nAct;
    
    counters_samp(indx_n0_y0:len_count:end) = N_n0_nAct;
    counters_samp(indx_n0_y1:len_count:end) = N_n0_Act;
    counters_samp(indx_d0_y0:len_count:end) = N_d0_nAct;
    counters_samp(indx_d0_y1:len_count:end) = N_d0_Act;
    
end
% 
%save([str,'_burnin.mat']);


beta_samples = zeros([p+1,N_samples]);
gamma_d0_samples = zeros([p+1,N_samples]);
gamma_n_samples = zeros([p+1,N_samples]);

gamma_d1_samples = sampleProbReg_Approx(X(1:nX,:),wg_d1(nX+1:end),wg_d1(1:nX)+wg_d1(nX+1:end),N_samples); % theta_d1 = P(Y=1|D=1,Z=1) -- Fixed during the iterations
prop =0.1;
k=1;

%--------------------------Actual Samples---------------------------
for s=1:N_samples
    if s>=round(prop*k*N_samples)
        k = k+1;
        fprintf('Drawing sample %d of %d\n',s,N_samples);
    end
    wg_d = [counters_samp(indx_n1_y0:len_count:end)+counters_samp(indx_n1_y1:len_count:end)+...
        counters_samp(indx_n0_y0:len_count:end)+counters_samp(indx_n0_y1:len_count:end);
        counters_samp(indx_d1_y0:len_count:end)+counters_samp(indx_d1_y1:len_count:end)+...
        counters_samp(indx_d0_y0:len_count:end)+counters_samp(indx_d0_y1:len_count:end)];
    wg_d = wg_d + alpha_0;
    beta_s = sampleProbReg_Approx(X(1:nX,:),wg_d(nX+1:end),wg_d(1:nX)+wg_d(nX+1:end),block);
    
    gamma_d1 = gamma_d1_samples(:,s);

    wg_d0 = [counters_samp(indx_d0_y0:len_count:end);counters_samp(indx_d0_y1:len_count:end)];
    wg_d0 = wg_d0 + alpha_0;
    gamma_d0 = sampleProbReg_Approx(X(1:nX,:),wg_d0(nX+1:end),wg_d0(1:nX)+wg_d0(nX+1:end),block);

    wg_n = [counters_samp(indx_n0_y0:len_count:end)+counters_samp(indx_n1_y0:len_count:end);...
        counters_samp(indx_n0_y1:len_count:end)+counters_samp(indx_n1_y1:len_count:end)];
    wg_n = wg_n + alpha_0;
    gamma_n = sampleProbReg_Approx(X(1:nX,:),wg_n(nX+1:end),wg_n(1:nX)+wg_n(nX+1:end),block);
    
    % X partition by half because X is doubled, one half for y=0 and the
    % other for y=1 in the gamma's sampling
    omega = normcdf(X(1:size(X,1)/2,:)*beta_s); % old omega
    theta_d0 = normcdf(X(1:size(X,1)/2,:)*gamma_d0); % old theta_d0
    theta_n = normcdf(X(1:size(X,1)/2,:)*gamma_n); %old theta_n

    Pd_d0_Act = omega.*theta_d0./(omega.*theta_d0+(1-omega).*theta_n);
    Pd_d0_nAct = omega.*(1-theta_d0)./(omega.*(1-theta_d0)+(1-omega).*(1-theta_n));

    N_d0_Act = binornd(counters_trans(indx_n0_y1:len_count:end,end),Pd_d0_Act);
    N_d0_nAct = binornd(counters_trans(indx_n0_y0:len_count:end,end),Pd_d0_nAct);
    N_n0_Act = counters_trans(indx_n0_y1:len_count:end,end) - N_d0_Act;
    N_n0_nAct = counters_trans(indx_n0_y0:len_count:end,end) - N_d0_nAct;
    
    counters_samp(indx_n0_y0:len_count:end) = N_n0_nAct;
    counters_samp(indx_n0_y1:len_count:end) = N_n0_Act;
    counters_samp(indx_d0_y0:len_count:end) = N_d0_nAct;
    counters_samp(indx_d0_y1:len_count:end) = N_d0_Act;

    % Output regression vector samples Eq 14
    beta_samples(:,s) = beta_s;
    gamma_d0_samples(:,s) = gamma_d0;
    gamma_n_samples(:,s) = gamma_n;

end
%save([str,'_full.mat']);


