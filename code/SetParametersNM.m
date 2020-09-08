% Set parameters for the 6th order JR model
%
function params = SetParametersNM(mode)

if nargin<1
    mode = 'alpha';
end

switch mode
    case 'alpha'
        
        params.e0 = 2.5;  % max firing rate
%         params.r = 0.56;  % logistic sigmoid
        params.r = 3;  % erf sigmoid
        params.v0 = 6;

        % inverse time constants
        params.a = 100;% (1/ tau_e)
        params.b = 50; % (1/tau_i)

        params.A = 3.25;  % gains (A = excitatory)
%         params.A = 5;
        params.B = 22;


        params.mu = 11;        % mean input mem potential.

        
end
