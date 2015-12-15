
% Algorithm 1 implementation.
% Gibbs Sampling Algorithm based on the joint distribution of Eq. 5
%
% Table 3 data input:
%
% Row 6: outMat(6) = N_{1S}^1
% Row 5: outMat(5) = N_{1S}^0
% Row 4: outMat(4) = N_{0S}^1
% Row 3: outMat(3) = N_{0S}^0
% Row 2: outMat(2) = N_{{0,1}C}^1
% Row 1: outMat(1) = N_{{0,1}C}^0
%
function [omega_samples,theta_d1_samples,theta_d0_samples,theta_n_samples]=...
    CE_mixture_init(outMat_vect,N_burnin,N_samples,alpha_0,beta_0)


theta_n_hat = outMat_vect(4)/sum(outMat_vect(3:4));
theta_d1_hat = outMat_vect(6)/sum(outMat_vect(5:6));
pd = sum(outMat_vect(5:6))/sum(outMat_vect(3:6));
p0_camp = sum(outMat_vect(2))/sum(outMat_vect(1:2));
theta_d0_hat = (p0_camp - theta_n_hat*(1-pd))/pd;

theta_d1 = theta_d1_hat;
theta_d0 = theta_d0_hat;
theta_n = theta_n_hat;
omega = pd;

% Initialization of class label for control
N_d1 = sum(outMat_vect(5:6));
N_n1 = sum(outMat_vect(3:4));
N_n1_Act = outMat_vect(4);
N_n1_nAct = outMat_vect(3);

Pd_d0_Act = omega*theta_d0/(omega*theta_d0+(1-omega)*theta_n);
Pd_d0_nAct = omega*(1-theta_d0)/(omega*(1-theta_d0)+(1-omega)*(1-theta_n));

N_d0_Act = binornd(outMat_vect(2),Pd_d0_Act);
N_d0_nAct = binornd(outMat_vect(1),Pd_d0_nAct);
N_n0_Act = outMat_vect(2) - N_d0_Act;
N_n0_nAct = outMat_vect(1) - N_d0_nAct;

%------------------------

% N_burnin = 500;
% N_samples = 3000;

prop =0.5;
k=1;

%----------Burnin--------------------------------------------
for s=1:N_burnin
    if s>=round(prop*k*N_burnin)
        k = k+1;
        s
    end
    alpha_omega = alpha_0+N_d1+N_d0_Act+N_d0_nAct ;
    beta_omega = beta_0+N_n1+N_n0_Act+N_n0_nAct;
    omega = betarnd(alpha_omega,beta_omega);    % omega = prob of qualified P(D=1)

    alpha_d1 = alpha_0+outMat_vect(6);
    beta_d1 = beta_0+outMat_vect(5);
    theta_d1 = betarnd(alpha_d1,beta_d1);          % theta_d1 = P(Y=1|D=1,Z=1) -- Fixed during the iterations

    alpha_d0 = alpha_0+N_d0_Act;
    beta_d0 = beta_0+N_d0_nAct;
    theta_d0 = betarnd(alpha_d0,beta_d0);          % theta_d0 = P(Y=1|D=1,Z=0)

    alpha_n = alpha_0+N_n0_Act+N_n1_Act;
    beta_n = beta_0+N_n0_nAct+N_n1_nAct;
    theta_n = betarnd(alpha_n,beta_n);             % theta_n = P(Y=1|D=0,Z=0) = P(Y=1|D=0,Z=1) = P(Y=1|D=0)

    Pd_d0_Act = omega*theta_d0/(omega*theta_d0+(1-omega)*theta_n);
    Pd_d0_nAct = omega*(1-theta_d0)/(omega*(1-theta_d0)+(1-omega)*(1-theta_n));

    N_d0_Act = binornd(outMat_vect(2),Pd_d0_Act);
    N_d0_nAct = binornd(outMat_vect(1),Pd_d0_nAct);
    N_n0_Act = outMat_vect(2) - N_d0_Act;
    N_n0_nAct = outMat_vect(1) - N_d0_nAct;
end

omega_samples = zeros([N_samples,1]);
theta_d1_samples = zeros([N_samples,1]);
theta_d0_samples = zeros([N_samples,1]);
theta_n_samples = zeros([N_samples,1]);

prop =0.1;
k=1;

%--------------------------Actual Samples---------------------------
for s=1:N_samples
    if s>=round(prop*k*N_samples)
        k = k+1;
        s
    end
    alpha_omega = alpha_0+N_d1+N_d0_Act+N_d0_nAct ;
    beta_omega = beta_0+N_n1+N_n0_Act+N_n0_nAct;
    omega = betarnd(alpha_omega,beta_omega);    % omega = prob of qualified P(D=1)

    alpha_d1 = alpha_0+outMat_vect(6);
    beta_d1 = beta_0+outMat_vect(5);
    theta_d1 = betarnd(alpha_d1,beta_d1);          % theta_d1 = P(Y=1|D=1,Z=1) -- Fixed during the iterations

    alpha_d0 = alpha_0+N_d0_Act;
    beta_d0 = beta_0+N_d0_nAct;
    theta_d0 = betarnd(alpha_d0,beta_d0);          % theta_d0 = P(Y=1|D=1,Z=0)

    alpha_n = alpha_0+N_n0_Act+N_n1_Act;
    beta_n = beta_0+N_n0_nAct+N_n1_nAct;
    theta_n = betarnd(alpha_n,beta_n);             % theta_n = P(Y=1|D=0,Z=0) = P(Y=1|D=0,Z=1) = P(Y=1|D=0)

    Pd_d0_Act = omega*theta_d0/(omega*theta_d0+(1-omega)*theta_n);
    Pd_d0_nAct = omega*(1-theta_d0)/(omega*(1-theta_d0)+(1-omega)*(1-theta_n));

    N_d0_Act = binornd(outMat_vect(2),Pd_d0_Act);
    N_d0_nAct = binornd(outMat_vect(1),Pd_d0_nAct);
    N_n0_Act = outMat_vect(2) - N_d0_Act;
    N_n0_nAct = outMat_vect(1) - N_d0_nAct;
    
    omega_samples(s) = omega;
    theta_d1_samples(s) = theta_d1;
    theta_d0_samples(s) = theta_d0;
    theta_n_samples(s) = theta_n;
end

