% This code repricates the structure of the Voss / Schiff FN simulation /
% estimation
%

clear all
close all
clc


N = 1000;             	% number of samples
dT = 0.001;          	% sampling time step (global)
dt = 1*dT;            	% integration time step
nn = fix(dT/dt);      	% (used in for loop for forward modelling) the integration time step can be small that the sampling (fix round towards zero)

t = 0:dt:(N-1)*dt;

% Intial true parameter values
%
parameters = SetParametersJR('seizure');
parameters.dt = dt;
A=parameters.A;
a=parameters.a;
sigma=parameters.sigma;

% Initialise random number generator for repeatability
%
rng(0);

% Transition model
%
NStates = 6;                           
f = @(x)model_JR(x,'transition',parameters);
F = @(x)model_JR(x,'jacobian',parameters);

% Initialise trajectory state
%
x0 = zeros(NStates,1);                   % initial state
x = zeros(NStates,N); 
x(:,1) = x0;

Q = (sqrt(dt)*A*a*sigma)^2*eye(NStates);
R = 0.01^2*eye(1);

% ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
% Generate trajectory
% ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

% Define input
%
e = sqrt(dt)*A*a*sigma*randn(N,1);

% Euler-Maruyama integration
%
for n=1:N-1
    x(:,n+1) = f(x(:,n))  + [0; 0; 0; e(n); 0; 0];
end

H = [0 0 1 0 -1 0];           % observation function
y = H*x;

% ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
% Run EKF
% ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

% Prior distribution (defined by m0 & P0)
%
m0 = x0;
P0 = F(x0)*Q*F(x0)';
P0=Q;

% Apply EKF filter
%
m = extended_kalman_filter(y,f,F,H,Q,R,m0,P0);

figure
plot(t,x([1 3 5],:)'); hold on;
plot(t,m([1 3 5],:)','--');

