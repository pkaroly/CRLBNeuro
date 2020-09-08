% the input is a deterministic sinusoidal firing rate + a stochastic input

function varargout = model_NM(x,P,mode,params)

% x = states or estimated states
% P = [] or covariance of estimated states


% the parameters
dt = params.dt;

e_0 = params.e0;
r = params.r;       % varsigma
v0 = params.v0;

a = params.a;
b = params.b;

A = params.A;  % synaptic gains
B = params.B;

u = params.mu;        % mean input firing rate.

C1 = 100;    % number of synapses
C2 = 100;

% the states
v_e = x(1);
z_e = x(2);

v_i = x(3);
z_i = x(4);

% the state covariance
if ~isempty(P)
    sigma_sq_e = P(1,1);  % excitatory
    sigma_sq_i = P(3,3);    % inhib
    
    cov_e_i = P(1,3);
    cov_i_e = P(3,1);
end

% Linear component of model
%
F = [1, dt, 0, 0; ...
    -b^2*dt, 1-2*b*dt, 0, 0; ...
    0, 0, 1, dt; ...
    0, 0, -a^2*dt, 1-2*a*dt];

% Sigmoid functions
%
% f_i = 1 ./ (1 + exp(r*(v0 - (mu - v_i))));     % inhibitory population firing rate
% f_e = 1 ./ (1 + exp(r*(v0 - v_e)));            % excitatory population firing rate
f_i = 0.5*erf((u - v_i - v0) / (sqrt(2) * r)) + 0.5;       % inhibitory population firing rate
f_e = 0.5*erf((v_e - v0) / (sqrt(2) * r)) + 0.5;             % excitatory population firing rate

switch mode
    
    case 'transition'
        
        % Nonlinear component
        %
        alpha_i = B*C2*2*e_0; % lumped constant
        alpha_e = A*C1*2*e_0;
        gx = [0; ...
            dt*alpha_e*b*f_i; ...
            0; ...
            dt*alpha_i*a*f_e];
        
        
        % Nonlinear transition model
        %
        varargout{1} = F*x + gx ;
        
        %     % input from inbib
        %     v_e_tplus1 = z_e*dt + v_e;
        %     z_e_tplus1 = (2*e_0*B*b*C2*f_i - 2*b*z_e - b^2*v_e)*dt + z_e;
        %
        %     % input from excit
        %     v_i_tplus1 = z_i*dt + v_i;
        %     z_i_tplus1 = (2*e_0*A*a*C1*f_e - 2*a*z_i - a^2*v_i)*dt + z_i;
        %
        %
        %     % % output
        %     out2 = [v_e_tplus1 z_e_tplus1 v_i_tplus1 z_i_tplus1]';
        %     assert(norm(out-out2)<1e-10)
        
    case 'analytic'
        % the expectation and covariance of a Gaussian distribution
        % transformed by the NMM
        
        % E[Fx + g(x)] = F*E[x] + E[g(x)]
        
        % expectations for inhibitory and excitatory firing rates
        % these erf inputs get re-used (boilerplate)
        input_i = (u - v_i - v0) / (sqrt(2 * (r + sigma_sq_i)));
        input_e = (v_e - v0) / (sqrt(2 * (r + sigma_sq_e)));
        
        e_gi = 0.5*erf(input_i) + 0.5;       % inhibitory population firing rate E[g(x_i)]
        e_ge = 0.5*erf(input_e) + 0.5;     % excitatory population firing rate E[g(x_e)]
        
        %
        % Nonlinear component of expectation
        %
        E_gx = [0; ...
            dt*B*b*C2*2*e_0*e_gi; ...
            0; ...
            dt*A*a*C1*2*e_0*e_ge];
        
        % Expectations for nonlinear components of covariance
        % need to fill out (by expanding the covariance term and using the
        % appropriate solutions)
        
        % f(x) = Fx + g(x)
        
        % P = E[(f(x) - mu)(f(x) - mu)]
        % mu = E[f(x)] = analytic_mean
        % E[(Fx + g(x) - E[Fx + g(x)])(...)]
        %
        
        %         E_x_gx =   % this has a solution E[x_e *  g(x_i)]
        %         E_gx_gx =  % this uses the Gaussian CDF  E[g(x)g(x)]
        %         mvncdf()
        
        % Analytic mean
        %
        analytic_mean = F*x + E_gx ;
        
        analytic_cov = F * P * F'; % + a bunch of other terms from expanding the cov expression...
        
        varargout{1} = analytic_mean;
        varargout{2} = analytic_cov;
        
    case 'jacobian'
        % Linearise g()
        %
        
        f_i_derivative = 2*e_0*r*f_i*(1-f_i);      % inhibitory feedback firing
        f_e_derivative = 2*e_0*r*f_e*(1-f_e);      % excitatory feedback firing
        
        G = [0, 0, 0, 0; ...
            0,0,-dt*B*b*C2*f_i_derivative,0; ...
            0, 0, 0, 0; ...
            dt*A*a*C1*f_e_derivative, 0, 0, 0];
        
        % Jacobian
        %
        varargout{1} = F + G;
end
