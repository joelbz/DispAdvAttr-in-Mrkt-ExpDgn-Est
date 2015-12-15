
% Simulate data counts (Table 3) given a set of model parameters 
% used to perform the power analysis of Appendix D.
% The count set generation is based ont he methond of moments of Eq 19

function dat_synth=simul_data(p0_d0,p0_d1,MCL,pd,pz,Nt)
% Inputs for simulation

p1_d1 = p0_d1*(MCL + 1); 
p0_camp = p0_d1*pd + p0_d0*(1-pd);

dat_synth = zeros([6,1]);
Ns = binornd(Nt,pz);
Nc = Nt - Ns;
dat_synth(2) = binornd(Nc,p0_camp);
dat_synth(1) = Nc - dat_synth(2);

Ns_d = binornd(Ns,pd);
dat_synth(6) = binornd(Ns_d,p1_d1);
dat_synth(5) = Ns_d - dat_synth(6);

dat_synth(4) = binornd(Ns-Ns_d,p0_d0);
dat_synth(3) = Ns-Ns_d - dat_synth(4);
