clear

filename = 'Fig6_moments_lift';
pd = 0.3;
%pd_samp = 0.1:0.1:0.9;
%MCL = 0.0513;
MCL_samp = 0.5:0.3:0.20;
p0_d1 = 1.48e-3;
p1_d1 = 1.56e-3;
p0_d0 = 1e-3;
%pz = 0.95;
pz_samp = 0.95:-0.03:0.85;
Nt = 37158296;
%Nt_samp = 5e6:5e6:40e6;

N_samples = 1e4;

p0_d0_samples = zeros(1,N_samples);
p0_d1_samples = zeros(1,N_samples);
MCL_samples = zeros(1,N_samples);
pd_samples = zeros(1,N_samples);

dat_synth_samples = zeros(6,N_samples);

for i=1:length(MCL_samp)
    MCL = MCL_samp(i);
    prop =0.01;
    k=1;
for m=1:length(pz_samp)
   pz=pz_samp(m);
for s=1:N_samples
    if s>=round(prop*k*N_samples)
        k = k+1;
        s
    end    
    dat_synth=simul_data(p0_d0,p0_d1,MCL,pd,pz,Nt);
    dat_synth_samples(:,s) = dat_synth;
    [p0_d0_samples(s),p0_d1_samples(s),MCL_samples(s),pd_samples(s),pz_,Nt_]=pointEst_moments(dat_synth);
end
save([filename,'_pd_',num2str(pd),'_pz_',num2str(pz),'.mat'])
[filename,'_pd_',num2str(pd),'_pz_',num2str(pz),'.mat']
end
end


