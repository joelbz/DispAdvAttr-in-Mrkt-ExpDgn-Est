
% If tested only for cr_prior = 0.5, this code replicates the results of Table 10
% Campaign 3 data
counters = [5492247,8131,9817552,1182,3713430,583,9938896,1246,3618467,607];
nameFile = 'Camp3_priorAnal_full';

% cr_prior = [0.5,0.01,1e-3];
cr_prior = 0.5;
ss_prior = 1;

cr_prior_full = repmat(cr_prior,1,length(ss_prior));
ss_prior_full = repmat(ss_prior,length(cr_prior),1);
ss_prior_full = ss_prior_full(:)';

alpha_0_test = cr_prior_full.*ss_prior_full;
beta_0_test = (1-cr_prior_full).*ss_prior_full;

len_runs = length(cr_prior)*length(ss_prior);
alpha_0 = 0.5;
beta_0 = 0.5;
N_burnin = 2000;
N_samples = 10000;

% ----------- Eq 7 Metrics ------------------
conf_ATE_D1 = zeros(len_runs,3);
conf_lift_D1 = zeros(len_runs,3);

% ----------- Eq 1 Metrics and lifts ------------------
conf_ATE_ad = zeros(len_runs,3);
conf_ACL_ad = zeros(len_runs,3);
conf_ATE_market = zeros(len_runs,3);
conf_ACL_market = zeros(len_runs,3);

% ----------- PSA vs Camp related (Eq 13 Statistics, Assumptions 1 and 2) ---------------------
conf_Delta_select = zeros(len_runs,3);
conf_Delta_convert = zeros(len_runs,3);
conf_Delta_select_lift = zeros(len_runs,3);
conf_Delta_convert_lift = zeros(len_runs,3);

% PSA (Placebo Campaign) related vectors
conf_theta_d0 = zeros(len_runs,3);
conf_theta_d1 = zeros(len_runs,3);
conf_theta_n = zeros(len_runs,3);
conf_omega = zeros(len_runs,3);

for i=1:len_runs
    i
    alpha_0 = alpha_0_test(i);
    beta_0 = beta_0_test(i);
    
    % ----------- Sampling for the Placebo arm ------------------

    alpha_d1 = alpha_0+counters(8);
    beta_d1 = beta_0+counters(7);
    theta_d1_PSA = betarnd(alpha_d1*ones([N_samples,1]),beta_d1*ones([N_samples,1]));          % theta_d1 = P(Y=1|D=1,Z=P)
    
    alpha_omega = alpha_0+sum(counters(7:8));
    beta_omega = beta_0+sum(counters(5:6));
    omega_PSA = betarnd(alpha_omega*ones([N_samples,1]),beta_omega*ones([N_samples,1]));    % omega = prob of qualified P(D=1|Z=P)

    alpha_n = alpha_0+counters(6);
    beta_n = beta_0+counters(5);
    theta_n_PSA = betarnd(alpha_n*ones([N_samples,1]),beta_n*ones([N_samples,1]));          % theta_n = P(Y=1|D=0,Z=P)

    % ----------- Running Algorithm 1 for the Control/Campaign treatment arms ------------------

    data_orig = [counters(1:2);counters(9:12)];
    [omega_samples,theta_d1_samples,theta_d0_samples,theta_n_samples]=...
    CE_mixture_init(data_orig,N_burnin,N_samples,alpha_0,beta_0);

    sortDiff = sort(theta_d0_samples);
    conf_theta_d0(i,:) = [sortDiff(round(0.05*N_samples)),median(sortDiff),sortDiff(round(0.95*N_samples))];
    
    sortDiff = sort(theta_d1_samples);
    conf_theta_d1(i,:) = [sortDiff(round(0.05*N_samples)),median(sortDiff),sortDiff(round(0.95*N_samples))];

    sortDiff = sort(theta_n_samples);
    conf_theta_n(i,:) = [sortDiff(round(0.05*N_samples)),median(sortDiff),sortDiff(round(0.95*N_samples))];
    
    sortDiff = sort(omega_samples);
    conf_omega(i,:) = [sortDiff(round(0.05*N_samples)),median(sortDiff),sortDiff(round(0.95*N_samples))];

    % ----------- Eq 7 Metrics ------------------

    sortDiff = sort(theta_d1_samples - theta_d0_samples);
    conf_ATE_D1(i,:) = [sortDiff(round(0.05*N_samples)),median(sortDiff),sortDiff(round(0.95*N_samples))];

    sortDiff = sort((theta_d1_samples - theta_d0_samples)./theta_d0_samples);
    conf_lift_D1(i,:) = [sortDiff(round(0.05*N_samples)),median(sortDiff),sortDiff(round(0.95*N_samples))];

    % ----------- Eq 1 Metrics ------------------

    sortDiff = sort(theta_d1_samples - theta_d1_PSA);
    conf_ATE_ad(i,:) = [sortDiff(round(0.05*N_samples)),median(sortDiff),sortDiff(round(0.95*N_samples))];

    sortDiff = sort(theta_d1_PSA - theta_d0_samples);
    conf_ATE_market(i,:) = [sortDiff(round(0.05*N_samples)),median(sortDiff),sortDiff(round(0.95*N_samples))];

    % ----------- Lift Eq 1 Metrics ------------------

    sortDiff = sort((theta_d1_samples - theta_d1_PSA)./theta_d1_PSA);
    conf_ACL_ad(i,:) = [sortDiff(round(0.05*N_samples)),median(sortDiff),sortDiff(round(0.95*N_samples))];

    sortDiff = sort((theta_d1_PSA - theta_d0_samples)./theta_d0_samples);
    conf_ACL_market(i,:) = [sortDiff(round(0.05*N_samples)),median(sortDiff),sortDiff(round(0.95*N_samples))];    

    % ----------- PSA vs Camp related (Eq 13 Statistics) ---------------------
    % Delta^select (Assumption 1, Eq 13)
    sortDiff = sort(omega_samples - omega_PSA);
    conf_Delta_select(i,:) = [sortDiff(round(0.05*N_samples)),median(sortDiff),sortDiff(round(0.95*N_samples))];

    sortDiff = sort((omega_samples - omega_PSA)./omega_PSA);
    conf_Delta_select_lift(i,:) = [sortDiff(round(0.05*N_samples)),median(sortDiff),sortDiff(round(0.95*N_samples))];

    % Delta^convert (Assumption 2, Eq 13)
    sortDiff = sort(theta_n_samples - theta_n_PSA);
    conf_Delta_convert(i,:) = [sortDiff(round(0.05*N_samples)),median(sortDiff),sortDiff(round(0.95*N_samples))];

    sortDiff = sort((theta_n_samples - theta_n_PSA)./theta_n_PSA);
    conf_Delta_convert_lift(i,:) = [sortDiff(round(0.05*N_samples)),median(sortDiff),sortDiff(round(0.95*N_samples))];

    % ---------- Targeting Evaluation Metrics (Section 3.3) ------------

   [Pd1_CE_pos,Pd1_CE_neg,Pd1_NCE_pos,Pd1_NCE_neg, PCE_pos_int,PCE_neg_int,PNCE_pos_int,PNCE_neg_int,CDT,CRDT]=...
    target_eval_func_marg(theta_d0_samples,theta_d1_samples,theta_n_samples,omega_samples);
    
    save([nameFile,'_Result_priorCR_',num2str(cr_prior_full(i)),'_priorSS_',num2str(ss_prior_full(i)),'.mat'])
end

save([nameFile,'_Result_done.mat'])
