
% Data Campaign 1
outMat_vect = [1560146;400;12010058;2387;5708558;2599];
nameFile = 'Camp1_priorAnal';

% Data Campaign 2
% outMat_vect = [2803640;734;18681097;3170;2584728;2685];
% nameFile = 'Camp2_priorAnal';

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

conf_ATE_D1 = zeros(len_runs,3);
conf_lift_D1 = zeros(len_runs,3);

conf_theta_d0 = zeros(len_runs,3);
conf_theta_d1 = zeros(len_runs,3);
conf_theta_n = zeros(len_runs,3);
conf_omega = zeros(len_runs,3);

for i=1:len_runs
    i

    alpha_0 = alpha_0_test(i);
    beta_0 = beta_0_test(i);
    
    %-------------------------------

    data_orig = outMat_vect;
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

    % ---------- Targeting Evaluation Metrics (Section 3.3) ------------

   [Pd1_CE_pos,Pd1_CE_neg,Pd1_NCE_pos,Pd1_NCE_neg, PCE_pos_int,PCE_neg_int,PNCE_pos_int,PNCE_neg_int,CDT,CRDT]=...
    target_eval_func_marg(theta_d0_samples,theta_d1_samples,theta_n_samples,omega_samples);

    save([nameFile,'_Result_priorCR_',num2str(cr_prior_full(i)),'_priorSS_',num2str(ss_prior_full(i)),'.mat'])
end

save([nameFile,'_Result_done.mat'])
