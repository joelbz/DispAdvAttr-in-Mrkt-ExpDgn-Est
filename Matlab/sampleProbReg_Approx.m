function beta_samples = sampleProbReg_Approx(X,ybin,nbin,numSamples)

[beta_hat,~,stats] = glmfit(X(:,2:end),[ybin nbin],'binomial','link','probit');

beta_samples = mvnrnd(beta_hat,stats.covb,numSamples)';
