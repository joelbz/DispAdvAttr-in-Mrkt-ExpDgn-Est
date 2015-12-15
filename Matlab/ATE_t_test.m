
% Function to estimate the confidence intervals for ATE for Bernoulli likelihood
% 
function [bound,ATE,lift,py_d0] = ATE_t_test(counters_St,counters_Ct)

% Assuming ATE= Ey_St - Ey_Ct
% counters_St = [#Converters, #Non-Converters] Study
% counters_Ct = [#Converters, #Non-Converters] Control

py_d1 = counters_St(2)./(counters_St(1)+counters_St(2));
var_d1 = py_d1.*(1-py_d1)./(counters_St(1)+counters_St(2));
py_d0 = counters_Ct(2)/(counters_Ct(1)+counters_Ct(2));
var_d0 = py_d0.*(1-py_d0)./(counters(4)+counters(3));

ATE = py_d1 - py_d0;
bound = norminv(0.95).*sqrt(var_d1+var_d0);

lift = ATE/py_d0;
