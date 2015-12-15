
% Method of moments point estimate (Eq 19 Appendix C) 
function [p0_d0,p0_d1,MCL,pd,pz,Nt]=pointEst_moments(dat_orig)

p0_d0 = sum(dat_orig(4))/sum(dat_orig(3:4));
p1_d1 = sum(dat_orig(6))/sum(dat_orig(5:6));
pd = sum(dat_orig(5:6))/sum(dat_orig(3:6));
p0_camp = sum(dat_orig(2))/sum(dat_orig(1:2));
p0_d1 = (p0_camp - p0_d0*(1-pd))/pd;
MCL = (p1_d1 - p0_d1)/p0_d1;

Nt = sum(dat_orig);
pz = sum(dat_orig(3:6))/sum(dat_orig(1:6));

