% Campaign effect on the overall visiting users, Eq 6
% Table 3 data input:
%
% Row 6: outMat(6) = N_{1S}^1
% Row 5: outMat(5) = N_{1S}^0
% Row 4: outMat(4) = N_{0S}^1
% Row 3: outMat(3) = N_{0S}^0
% Row 2: outMat(2) = N_{{0,1}C}^1
% Row 1: outMat(1) = N_{{0,1}C}^0

%---------------------------------------------%
% Bayesian Analysis

N_samples = 1e6;
alph_prior1 = 0.5;
beta_prior1 = 0.5;
alph_prior0 = 0.5;
beta_prior0 = 0.5;

alpha1 = alph_prior1 + outMat_vect(4) + outMat_vect(6);
beta1 = beta_prior1 + n1 - (outMat_vect(4) + outMat_vect(6));
p1_b=[betainv(0.05,alpha1,beta1),alpha1/(alpha1+beta1),betainv(0.95,alpha1,beta1)];

alpha0 = alph_prior0 + outMat_vect(2);
beta0 = beta_prior0 + n0 - outMat_vect(2);
p0_b=[betainv(0.05,alpha0,beta0),alpha0/(alpha0+beta0),betainv(0.95,alpha0,beta0)];

p0_samples = betarnd(alpha0,beta0,[N_samples,1]);
p1_samples = betarnd(alpha1,beta1,[N_samples,1]);

% CL campaign effect ATE_camp
% CD campaig effect difference 

CL_samples = sort((p1_samples - p0_samples)./p0_samples);
CD_samples = sort(p1_samples - p0_samples);

lowCL_b = CL_samples(round(0.05*N_samples));
medCL_b = median(CL_samples);
highCL_b = CL_samples(round(0.95*N_samples));

lowCD_b = CD_samples(round(0.05*N_samples));
medCD_b = median(CD_samples);
highCD_b = CD_samples(round(0.95*N_samples));

% Eq 8
																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																	
actAtt_int = [lowCD_b,medCD_b,highCD_b].*sum(outMat_vect(3:6))./(outMat_vect(6)+outMat_vect(4));
