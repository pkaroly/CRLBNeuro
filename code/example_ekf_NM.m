%% Simulate pendulum model and EKF estimates
%
close all

params = SetParametersNM('alpha');

params.dt = 0.001;


N = 5000;             	% number of samples
dT = params.dt;         % sampling time step (global)
dt = 1*dT;            	% integration time step
nn = fix(dT/dt);      	% (used in for loop for forward modelling) the integration time step can be small that the sampling (fix round towards zero)
t = 0:dt:(N-1)*dt;



% Transition model
NStates = 4;                           
f = @(x)model_NM(x,'transition',params);
F = @(x)model_NM(x,'jacobian',params);

% Initialise trajectory state
x0 = zeros(NStates,1);                   % initial state
x = zeros(NStates,N); 
x(:,1) = mvnrnd(x0,10^2*eye(NStates));

% ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
% Generate trajectory
% ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

% Euler integration
%
for n=1:N-1
    x(:,n+1) = f(x(:,n));
end

% Calculate noise covariance based on trajectory variance over time??
%   Why noise on all states?
%
Q = 10^2.*diag((0.4*std(x,[],2)*sqrt(dt)).^2);

% Initialise random number generator for repeatability
rng(0);

v = mvnrnd(zeros(NStates,1),Q,N)';

% Euler-Maruyama integration
for n=1:N-1
    x(:,n+1) = f(x(:,n)) + v(:,n);
end

H = [1 0 0 0];           % observation function
R = 1^2*eye(1);

w = mvnrnd(zeros(size(H,1),1),R,N)';
y = H*x + w;

%% Run EKF for this model

% Prior distribution (defined by m0 & P0)
%
m0 = x0;
P0 = 100.^2*eye(NStates);


% Apply EKF filter
%
m = extended_kalman_filter(y,f,F,H,Q,R,m0,P0);
%% Compute the posterior Cramer-Rao bound (PCRB)
%
M = 100;    % Number of Monte Carlo samples
pcrb = compute_pcrb_P(t,f,F,@(x)H,Q,R,m0,P0,M);
pcrbx5 = compute_pcrb_P(t,f,F,@(x)H,Q,R,m0,P0.*5,M); % Changed initial condition, multiply P0 by 5
pcrbd5 = compute_pcrb_P(t,f,F,@(x)H,Q,R,m0,P0./5,M); % Changed initial condition, divide P0 by 5

%% Plot
%
figure
ax1=subplot(211);
plot(t,x([1],:)'); hold on;
plot(t,m([1],:)','--');
legend({'Actual','Estimation'});
ax2=subplot(212);
plot(t,y)
legend('Observed EEG');

% plot(t,pcrb(1,:)')
% legend({'CRLB'})

linkaxes([ax1 ax2],'x');

%%
% return

%% Compute the MSE of the extended Kalman filter 
%
num_trials = 100;
error = zeros(NStates,N);

% parfor r=1:num_trials
for r=1:num_trials
    
    % Create new trajectory realisation
    %
    v = mvnrnd([0 0 0 0]',Q,N)';
    x = zeros(NStates,N);
    x(:,1)=mvnrnd(m0,P0)';
    for i=NStates:N
        x(:,i) = f(x(:,i-1)) + v(:,i);
    end

    % Generate new observations 
    %
    w = mvnrnd(zeros(1,size(H,1)),R,N)';
    z = H*x + w;

    % Apply EKF filter
    %
    m = extended_kalman_filter(z,f,F,H,Q,R,m0,P0);

    % Accumulate the estimation error
    %
    error = error + (x-m).^2;
end

% Calculate the mean squared error
%
mse = error ./ num_trials;
rmse = sqrt(error ./ num_trials);

%% Plot MSE and the PCRB
%
figure('Name', 'NMM - EKF vs CRB')
for i = 1:4
    subplot(2,2,i)
    semilogy(mse(i,:),'.-');
    hold on;
    %semilogy(rmse(i,:),'x-');
    semilogy(pcrb(i,:),'.-');
    grid on;
    legend({'MSE', 'PCRB'});
    xlabel('Time (s)');
    ylabel(['MSE state ' num2str(i) ' (mV)']);
    hold off;
end

%% Plot average MSE vs CRB 
figure('Name', 'Mean Vm - EKF vs CRB')
color = [0.1,0.6,0.7];
%semilogy(rmse(i,:),'x-');
semilogy(mean(mse,2),'o',...
    'MarkerSize',10,...
    'MarkerEdgeColor','k',...
    'MarkerFaceColor',color)
hold on;
bar(mean(pcrb,2), 'FaceColor', color+0.2);
semilogy(mean(mse,2),'o',...
    'MarkerSize',10,...
    'MarkerEdgeColor','k',...
    'MarkerFaceColor',color)
%yt = get(gca, 'YTick'); YTickLabels = cellstr(num2str(round(log10(yt(:))), '10^%d')); % Format y axis in log scale
grid on;
legend({'MSE', 'PCRB'});
xlabel('Time (s)');
ylabel('MSE (mV)');

%% Plot the evolution of the CRB
%
StateNum = 1;

pcrb_sqrt = movmean(sqrt(pcrb),1,2); % Sqrt of PCRB original initial conditions
pcrbx5_sqrt = movmean(sqrt(pcrbx5),1,2); % Sqrt of PCRB when P0 is 5 times bigger
pcrbd5_sqrt = movmean(sqrt(pcrbd5),1,2); % Sqrt of PCRB when P0 is 5 times smaller
range_ = 10:length(t)/2; % Range of values to plot to avoid the initial overshoot and to remove the entries beyond convergence

figure('Name', 'CRB Convergence')
plot(t(range_), pcrb_sqrt(StateNum, range_), 'LineWidth', 2);
hold on;
plot(t(range_), pcrbx5_sqrt(StateNum, range_), '--', 'LineWidth', 2);
plot(t(range_), pcrbd5_sqrt(StateNum, range_), '--', 'LineWidth', 2);
legend({'P0 = 10000I', 'P0 = 50000I', 'P0 = 2000I'});
grid on;
title(['State: ', num2str(StateNum)]);
xlabel('Time (s)');
ylabel('BCRB(mV)');
% xlim([0 5]);

%%
% figure
% semilogy(t,sum(mse),'x-')
% hold on;
% semilogy(t,sum(pcrb),'o-');
% grid on;
% legend({'MSE','PCRB'})
% xlabel('Time (s)');
% ylabel('MSE');