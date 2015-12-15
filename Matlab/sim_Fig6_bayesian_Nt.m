

filename = 'Fig6_bayesian_Nt';
pd = 0.3;
%pd_samp = 0.1:0.1:0.9;
MCL = 0.0513;
p0_d1 = 1.48e-3;
p1_d1 = 1.56e-3;
p0_d0 = 1e-3;
%pz = 0.95;
pz_samp = 0.95:-0.03:0.85;
%Nt = 37158296;
Nt_samp = 10e6:5e6:40e6;

N_burnin_CE = 2000;
N_samples_CE = 10000;

alpha_0 = 0.5;
beta_0 = 0.5;
for m=1:length(pz_samp)
   pz=pz_samp(m)
for j=1:length(Nt_samp)
    Nt = Nt_samp(j);
        
    countEst_sym = countEst_given_param(p0_d0,p0_d1,MCL,pd,pz,Nt);

    [omega_samples,theta_d1_samples,theta_d0_samples,theta_n_samples]=CE_mixture_init(countEst_sym,N_burnin_CE,N_samples_CE,alpha_0,beta_0);
    sortDiff = sort((theta_d1_samples - theta_d0_samples)./theta_d0_samples);
    MCL_int = [sortDiff(round(0.05*N_samples_CE)),median(sortDiff),sortDiff(round(0.95*N_samples_CE))];

    sortDiff = sort(theta_d1_samples - theta_d0_samples);
    MCE_int = [sortDiff(round(0.05*N_samples_CE)),median(sortDiff),sortDiff(round(0.95*N_samples_CE))];

    save([filename,num2str(alpha_0),'_',num2str(pz),'_',num2str(pd),'.mat'])    
    [filename,num2str(alpha_0),'_',num2str(pd),'.mat']
end
end

