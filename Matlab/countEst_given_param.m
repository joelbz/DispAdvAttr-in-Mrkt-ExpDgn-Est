
% We generate the counts of Table 3 assuming the point estimate from Eq 19 is perfect 
% given a set of model parameters used to perform the power analysis of Appendix D.

function dat_synth=countEst_given_param(p0_d0,p0_d1,MCL,pd,pz,Nt)
% Inputs for simulation

p1_d1 = p0_d1*(MCL + 1); 
p0_camp = p0_d1*pd + p0_d0*(1-pd);

dat_synth = zeros([6,1]);

Ns = Nt.*pz;
Nc = Nt - Ns;
dat_synth(2) = Nc.*p0_camp;
dat_synth(1) = Nc - dat_synth(2);

Ns_d = Ns.*pd;
dat_synth(6) = Ns_d.*p1_d1;
dat_synth(5) = Ns_d - dat_synth(6);

dat_synth(4) = (Ns-Ns_d).*p0_d0;
dat_synth(3) = Ns-Ns_d - dat_synth(4);

dat_synth = round(dat_synth);
