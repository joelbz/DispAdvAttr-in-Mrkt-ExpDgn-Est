

% -- Training in the first half of the campaign given results of Algorithm 2
% From algorithm 2: beta_samples, gamma_d0_samples, gamma_d1_sample, gamma_n_sample
%
% Variable mapping
%
% gamma_1C -> gamma_d0
% gamma_1S -> gamma_d1
% gamm_0 -> gamma_n
% beta_sel -> beta
% 
%
% -- Testing in the second half of the campaign
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


% Find the conversion probabilities of targeted and non-targeted for each user segment 

N_samples = size(beta_samples,2);

omega_samples = zeros(size(X,1)/2,N_samples)*-1;
theta_d0_samples = zeros(size(X,1)/2,N_samples)*-1;
theta_d1_samples = zeros(size(X,1)/2,N_samples)*-1;
theta_n_samples = zeros(size(X,1)/2,N_samples)*-1;
for s=1:N_samples
    omega_samples(:,s) = normcdf(X(1:size(X,1)/2,:)*beta_samples(:,s)); % old omega
    theta_d0_samples(:,s) = normcdf(X(1:size(X,1)/2,:)*gamma_d0_samples(:,s)); % old theta_d0
    theta_n_samples(:,s) = normcdf(X(1:size(X,1)/2,:)*gamma_n_samples(:,s)); %old theta_n
    theta_d1_samples(:,s) = normcdf(X(1:size(X,1)/2,:)*gamma_d1_samples(:,s)); % old theta_d1
end

% Find the targeting evaluation of Eq sec 3.3 for all user segments
Pd1_CE_pos = zeros(size(theta_d1_samples,1),3);
Pd1_CE_neg = zeros(size(theta_d1_samples,1),3);
Pd1_NCE_pos = zeros(size(theta_d1_samples,1),3);
Pd1_NCE_neg = zeros(size(theta_d1_samples,1),3);
PCE_pos_d1 = zeros(size(theta_d1_samples,1),3);
PCE_neg_d1 = zeros(size(theta_d1_samples,1),3);
PNCE_pos_d1 = zeros(size(theta_d1_samples,1),3);
PNCE_neg_d1 = zeros(size(theta_d1_samples,1),3);
for x=1:size(theta_d1_samples,1)
[Pd1_CE_pos(x,:),Pd1_CE_neg(x,:),Pd1_NCE_pos(x,:),Pd1_NCE_neg(x,:),...
    PCE_pos_d1(x,:),PCE_neg_d1(x,:),PNCE_pos_d1(x,:),PNCE_neg_d1(x,:)]=...
    target_eval_func(theta_d0_samples(x,:),theta_d1_samples(x,:),theta_n_samples(x,:),omega_samples(x,:),0);
end






% Estimating the inputs of Algorithm 3

% ----- F_sel(X_i) estimation --------------

% Table 12 targeting polices
% (a)
re_weights = PCE_pos_d1(:,2)./PCE_neg_d1(:,2);
% (b)
%re_weights = PCE_pos_d1(:,2)./(PCE_neg_d1(:,2)+PNCE_pos_d1(:,2));
% (c)
%re_weights = PCE_pos_d1(:,2)./(1-PCE_pos_d1(:,2));
% (d)
%re_weights = mean(theta_d1_samples,2)./(1-mean(theta_d1_samples,2));


% --------- Non-zero Effect Indicator function F_sig(X_i)

LATE_samples = (theta_d1_samples - theta_d0_samples);
LATE_samples = sort(LATE_samples,2);
LATE_med = median(LATE_samples,2);
LATE_low = LATE_samples(:,floor(N_samples*0.05));
LATE_high = LATE_samples(:,floor(N_samples*0.95));

significant = sign(LATE_low)==sign(LATE_high);

% --------- ATE^D=1_Camp Sign function F^ATE_sign(X_i) ----------------
significant_pos = sign(LATE_low)==sign(LATE_high) & LATE_low>0;
significant_neg = sign(LATE_low)==sign(LATE_high) & LATE_high<0;

w_sig = [1,1,1];
%w_sig = [0.6,1,1.1];
%w_sig = [0.8,1,1.2];
%w_sig = [0.8,1,1.1];
% ------ Actual Algorithm 3 -----------------

re_weights(significant_neg) = re_weights(significant_neg)*w(1);
re_weights(~significant) = re_weights(~significant)*w(2);
re_weights(significant_pos) = re_weights(significant_pos)*w(3);



% -- Testing in the second half of the campaign

% Data to test the user targeting (2nd half of the campaign)
% Call and get the data: counters_trans as in the case of Algorithm 2

indx_n0_y0 = 1;
indx_n0_y1 = 2;
indx_d0_y0 = 3;
indx_d0_y1 = 4;
indx_n1_y0 = 5;
indx_n1_y1 = 6;
indx_d1_y0 = 7;
indx_d1_y1 = 8;

% Testing the user targeting for the study group
counters_targ_marg = targetUsr([counters_trans(indx_n1_y0:4:end,:); counters_trans(indx_n1_y1:4:end,:);...
			counters_trans(indx_d1_y0:4:end,:);counters_trans(indx_d1_y1:4:end,:)],re_weights);

counters_marg(1) = sum(counters_trans(indx_n0_y0:len_count:end,end));
counters_marg(2) = sum(counters_trans(indx_n0_y1:len_count:end,end));
counters_marg(3:6)= counters_targ_marg;
counters_marg = round(counters_marg);

% Run Algorithm 1 to find the effect
N_samples = 10000;
N_burnin = 3000;
[omega_samples,theta_d1_samples,theta_d0_samples,theta_n_samples]=CE_mixture_init(counters_marg,N_burnin,N_samples,0.5,0.5);




