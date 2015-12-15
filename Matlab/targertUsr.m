% Algorithm 4 impletation
%
% Input:
%
% "counters_trans" data matrix where the columns 1:p are the feature vector X,
% last column of counters_trans represents the user count set 
%
% These features are repeated 4 times to capture each of the user counts of
% the study treatment arm
%
% F_targ: Targeting function


function counters_targ_marg = targetUsr(counters_trans,F_targ)

% Given X_i (i=0:feat_Combinations-1) and p vector size, we assume:
% counters_trans(i*4+1,p+1) = N_{0Z}^0
% counters_trans(i*4+2,p+1) = N_{0Z}^1
% counters_trans(i*4+3,p+1) = N_{1Z}^0
% counters_trans(i*4+4,p+1) = N_{1Z}^1

re_weights = F_targ;

p = size(counters_trans,2) -  1; 

% Data count indices given a feature vector value X_i
indx_n1_y0 = 1;
indx_n1_y1 = 2;
indx_d1_y0 = 3;
indx_d1_y1 = 4;

len_count = 4;


%--- 
counters_trans_targ = counters_trans;

% re-allocating users to consume the total budget
N_d1 = counters_trans(indx_d1_y1:len_count:end,end)+counters_trans(indx_d1_y0:len_count:end,end);
N_n1 = counters_trans(indx_n1_y1:len_count:end,end)+counters_trans(indx_n1_y0:len_count:end,end);

% Features are included in the columns of counters_trans_targ
X = counters_trans(1:len_count:end,1:p);
beta_d1_hat = glmfit(X,[counters_trans(indx_d1_y1:len_count:end,end) N_d1],'binomial','link','probit');
beta_n_hat = glmfit(X,[counters_trans(indx_n1_y1:len_count:end,end) N_n1],'binomial','link','probit');

py_d1_hat = normcdf([ones(size(counters_trans,1)/len_count,1),X]*beta_d1_hat);
py_n_hat = normcdf([ones(size(counters_trans,1)/len_count,1),X]*beta_n_hat);


%--- Re-allocating the exposed population in study----
N_1 = counters_trans(indx_n1_y1:len_count:end,end)+counters_trans(indx_n1_y0:len_count:end,end) + ...
    counters_trans(indx_d1_y1:len_count:end,end)+counters_trans(indx_d1_y0:len_count:end,end);
N_d1_y0 = N_1*0;
N_d1_y1 = N_1*0;
N_d1_rem = sum(N_d1);
while N_d1_rem>10
    N_1_rem = N_1 - (N_d1_y1+N_d1_y0);
    pX_hat = N_1_rem./sum(N_1_rem);

    norm_cte = N_d1_rem./sum(re_weights.*N_d1_rem.*pX_hat);
    weight_d1_new = norm_cte.*re_weights;
    N_d1_y0 = N_d1_y0+ min(weight_d1_new.*N_d1_rem.*pX_hat.*(1-py_d1_hat),(1-py_d1_hat).*N_1_rem);
    N_d1_y1 = N_d1_y1+ min(weight_d1_new.*N_d1_rem.*pX_hat.*py_d1_hat,py_d1_hat.*N_1_rem);
    
    N_d1_rem = sum(N_d1 - (N_d1_y0 + N_d1_y1));
end
counters_trans_targ(indx_d1_y0:len_count:end,end) = N_d1_y0;
counters_trans_targ(indx_d1_y1:len_count:end,end) = N_d1_y1;

%-- Getting conversion rates for the remaining population ------------

N_n1 = N_1 - (N_d1_y0 + N_d1_y1); % Minus those that are now targeted

counters_trans_targ(indx_n1_y1:len_count:end,end) = py_n_hat.*N_n1;
counters_trans_targ(indx_n1_y0:len_count:end,end) = (1-py_n_hat).*N_n1;

% -------- Aggregating counters ---------------------------------
counters_targ_marg(1) = sum(counters_trans_targ(indx_n1_y0:len_count:end,end));
counters_targ_marg(2) = sum(counters_trans_targ(indx_n1_y1:len_count:end,end));
counters_targ_marg(3) = sum(counters_trans_targ(indx_d1_y0:len_count:end,end));
counters_targ_marg(4) = sum(counters_trans_targ(indx_d1_y1:len_count:end,end));

