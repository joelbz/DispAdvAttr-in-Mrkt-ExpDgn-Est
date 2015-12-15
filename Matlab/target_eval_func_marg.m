% Function to generate the targeting evaluation metrics of section 3.3
% The input are the vectors of Gibb samples from Algorithm 1 (Eq 5)
% The output are the posterior intervals for the metrics of section 3.3
%
% Variable mapping
%
% Input:
%
% theta_1C -> theta_d0
% theta_1S -> theta_d1
% theta_0 -> theta_n
% p_sel -> pi
%
% Output:
%
% SelfEff-> CDT			Eq 9
% lift_sel-> CRDT		Eq 9
% P(D=1|U=Per+)-> Pd1_CE_pos	Eq 11
% P(D=1|U=Per-)-> Pd1_CE_neg	Eq 11
% P(D=1|U=AB)-> Pd1_NCE_pos	Eq 11
% P(D=1|U=NB)-> Pd1_NCE_neg	Eq 11
% P(U=Per+)-> PCE_pos_int	
% P(U=Per-)-> PCE_neg_int	
% P(U=AB)-> PNCE_pos_int	
% P(U=NB)-> PNCE_neg_int	

function [Pd1_CE_pos,Pd1_CE_neg,Pd1_NCE_pos,Pd1_NCE_neg, ...
    PCE_pos_int,PCE_neg_int,PNCE_pos_int,PNCE_neg_int,CDT,CRDT]=...
    target_eval_func_marg(theta_d0_samples,theta_d1_samples,theta_n_samples,pi_samples)

N_samples = length(theta_d0_samples);
% Ability to find high converters
% Average coversion difference due to the targeting engine
% CDT = E(Y|D=1,Z=0) - E(Y|D=0,Z=0)
% CDT = theta_10 - theta_0
CDT_samples = theta_d0_samples - theta_n_samples; 
sortDiff = sort(CDT_samples);
CDT = [sortDiff(round(0.05*N_samples)),median(sortDiff),sortDiff(round(0.95*N_samples))];


% Average coversion relative difference due to the targeting engine
% CRDT = (E(Y|D=1,Z=0) - E(Y|D=0,Z=0))/E(Y|D=0,Z=0)
CRDT_samples = (theta_d0_samples - theta_n_samples)./theta_n_samples;
sortDiff = sort(CRDT_samples);
CRDT = [sortDiff(round(0.05*N_samples)),median(sortDiff),sortDiff(round(0.95*N_samples))];


%%
% Analysis on the targeted population (conditional on D=1) Eq 10
PCE_pos_d1 = theta_d1_samples.*(1-theta_d0_samples);
PCE_neg_d1 = (1-theta_d1_samples).*theta_d0_samples;
PNCE_pos_d1 = theta_d1_samples.*theta_d0_samples;
PNCE_neg_d1 = (1-theta_d1_samples).*(1-theta_d0_samples);

normConst = PCE_pos_d1 + PCE_neg_d1 + PNCE_pos_d1 + PNCE_neg_d1;
PCE_pos_d1 = PCE_pos_d1./normConst;
PCE_neg_d1 = PCE_neg_d1./normConst;
PNCE_pos_d1 = PNCE_pos_d1./normConst;
PNCE_neg_d1 = PNCE_neg_d1./normConst;

sort_samples = sort(PCE_pos_d1);
PCE_pos_d1_int = [sort_samples(round(0.05*N_samples)),median(sort_samples),sort_samples(round(0.95*N_samples))];
sort_samples = sort(PCE_neg_d1);
PCE_neg_d1_int = [sort_samples(round(0.05*N_samples)),median(sort_samples),sort_samples(round(0.95*N_samples))];
sort_samples = sort(PNCE_pos_d1);
PNCE_pos_d1_int = [sort_samples(round(0.05*N_samples)),median(sort_samples),sort_samples(round(0.95*N_samples))];
sort_samples = sort(PNCE_neg_d1);
PNCE_neg_d1_int = [sort_samples(round(0.05*N_samples)),median(sort_samples),sort_samples(round(0.95*N_samples))];

% Switching back probabilities (Bayes Theorem) Eq 11
PCE_pos_d0 = theta_n_samples.*(1-theta_n_samples);
PCE_neg_d0 = (1-theta_n_samples).*theta_n_samples;
PNCE_pos_d0 = theta_n_samples.*theta_n_samples;
PNCE_neg_d0 = (1-theta_n_samples).*(1-theta_n_samples);

normConst = PCE_pos_d0 + PCE_neg_d0 + PNCE_pos_d0 + PNCE_neg_d0;
PCE_pos_d0 = PCE_pos_d0./normConst;
PCE_neg_d0 = PCE_neg_d0./normConst;
PNCE_pos_d0 = PNCE_pos_d0./normConst;
PNCE_neg_d0 = PNCE_neg_d0./normConst;

PCE_pos = pi_samples.*PCE_pos_d1 + (1-pi_samples).*PCE_pos_d0;
PCE_neg = pi_samples.*PCE_neg_d1 + (1-pi_samples).*PCE_neg_d0;
PNCE_pos = pi_samples.*PNCE_pos_d1 + (1-pi_samples).*PNCE_pos_d0;
PNCE_neg = pi_samples.*PNCE_neg_d1 + (1-pi_samples).*PNCE_neg_d0;

sort_samples = sort(PCE_pos);
PCE_pos_int = [sort_samples(round(0.05*N_samples)),median(sort_samples),sort_samples(round(0.95*N_samples))];
sort_samples = sort(PCE_neg);
PCE_neg_int = [sort_samples(round(0.05*N_samples)),median(sort_samples),sort_samples(round(0.95*N_samples))];
sort_samples = sort(PNCE_pos);
PNCE_pos_int = [sort_samples(round(0.05*N_samples)),median(sort_samples),sort_samples(round(0.95*N_samples))];
sort_samples = sort(PNCE_neg);
PNCE_neg_int = [sort_samples(round(0.05*N_samples)),median(sort_samples),sort_samples(round(0.95*N_samples))];



Pd1_CE_pos_samples = pi_samples.*PCE_pos_d1./(pi_samples.*PCE_pos_d1 + (1-pi_samples).*PCE_pos_d0);
Pd1_CE_neg_samples = pi_samples.*PCE_neg_d1./(pi_samples.*PCE_neg_d1 + (1-pi_samples).*PCE_neg_d0);
Pd1_NCE_pos_samples = pi_samples.*PNCE_pos_d1./(pi_samples.*PNCE_pos_d1 + (1-pi_samples).*PNCE_pos_d0);
Pd1_NCE_neg_samples = pi_samples.*PNCE_neg_d1./(pi_samples.*PNCE_neg_d1 + (1-pi_samples).*PNCE_neg_d0);


sort_samples = sort(Pd1_CE_pos_samples);
Pd1_CE_pos = [sort_samples(round(0.05*N_samples)),median(sort_samples),sort_samples(round(0.95*N_samples))];
sort_samples = sort(Pd1_CE_neg_samples);
Pd1_CE_neg = [sort_samples(round(0.05*N_samples)),median(sort_samples),sort_samples(round(0.95*N_samples))];
sort_samples = sort(Pd1_NCE_pos_samples);
Pd1_NCE_pos = [sort_samples(round(0.05*N_samples)),median(sort_samples),sort_samples(round(0.95*N_samples))];
sort_samples = sort(Pd1_NCE_neg_samples);
Pd1_NCE_neg = [sort_samples(round(0.05*N_samples)),median(sort_samples),sort_samples(round(0.95*N_samples))];



