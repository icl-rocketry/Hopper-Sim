function P = rocketConfigEuler()
%ROCKETCONFIG  All physical constants, gains, trajectory, *and ICs*
%body info
R = 0.3; %radius of vehicle
ISP = 180; %ISP of engine
%% ========== PHYSICAL CONSTANTS ==========================================
P.g      = 9.81;
P.rEng   = [0; 0; -1];

P.alpha1 = 1/(ISP*9.81);     % mdot coeff. = (ISP*gravity)^(-1) look at appendix B for info
P.alpha2 = (R^2)/(2*ISP*9.81);     % Jdot coeff. = (R^2 * 1/(2*ISP*gravity)) 
P.beta   = 0.03;      % throttle → flow-rate

%% ========== DISTURBANCE MODEL ===========================================
P.useDist          = false;                 % <-- set TRUE so the lines below take effect
% --- Force disturbance ---------------------------------------------------
%here you have the option
% Body-frame components:  Fx = 20 sin(3 t) , Fy = 0 , Fz = 20 sin(2 t)
P.distForceGlobal  = true;                % false → you're giving body-frame values
P.distForceFcn     = @(t) [ 20*sin(3*t) ; ...
                            0            ; ...
                            20*sin(2*t) ];
% --- Moment disturbance --------------------------------------------------
% Zero roll-, pitch- and yaw-torque
P.distMomentGlobal = false;                % remains in body frame (anyway zero)
P.distMomentFcn    = @(t) [ 20 ; 0 ; 0 ];

%% ================= INITIAL CONDITIONS  ==================================
% Local variables first (all column vectors)
pos0   = [0; 0; 0];          % 3×1
vel0   = [0; 0; 0];          % 3×1
eul0  =  [0; 0; 0];           % 3×1  (make it a column)
omega0 = [0; 0; 0];          % 3×1

% Optional mass / inertia
m0     = 35;                 % scalar
J0     = diag([3.5 3.5 0.447]);      % 3×3

% Build the full 12×1 state vector in one shot
state0 = [ pos0        ;    % 3
           vel0        ;    % 3
           eul0       ;    % 3
           omega0 ];        % 3

% --- after building state0 (12×1) ---
stateDot0 = zeros(size(state0));    % 12×1 zero derivative

initStruct = struct( ...
    'pos'   , pos0      , ...
    'vel'   , vel0      , ...
    'quat'  , eul0     , ...
    'omega' , omega0    , ...
    'mass'  , m0        , ...
    'J'     , J0        , ...
    'state' , state0    , ...
    'stateDot', stateDot0 );   % <— new line

P.init = initStruct;
%% ========== NOMINAL 3-D TRAJECTORY ======================================
P.traj.x0 = 0;   P.traj.xi = 2;   P.traj.xf = 4;
P.traj.y0 = 0;   P.traj.yi = 2;   P.traj.yf = 4;
P.traj.z0 = 0;   P.traj.zi = 2;   P.traj.zf = 0;
P.traj.t1       = 5;
P.traj.t2       = 5;
P.traj.dt_blend = 2;
P.traj.frac_tb  = 0.20;
% ------------ after your existing traj fields -------------
P.traj.useStep   = false;         % set true to enable a hard step
P.traj.stepTime  = 2.0;           % the time (s) at which the jump occurs
P.traj.stepTarget = [0.1; 0; 0];  % [x; y; z] you want after the step
% ----------------------------------------------------------

%% ========== OUTER-LOOP TRANSLATION GAINS ================================
P.kPos1          = [20  20  6];     % k₁ for x, y, z
P.kPos2          = [10 10 10];     % k₂ for x, y, z
P.lambdaPosOuter = [8  8  8];     % λ  for x, y, z
P.etaPosOuter    = 0.4;         % same η for all axes
P.psiPosOuter    = [5 5 0.1]; % ψ  per axis

%% ========== INNER-LOOP (Fz) GAINS =======================================
P.kPos3 = 10;   P.kPos4 = 10;
P.lambdaPosInner = 8;
P.etaPosInner    = 0.5;
P.psiPosInner    = 5;%dont set to 0, 

%% ========== ATTITUDE GAINS ==============================================
P.kAtt1 = 30;  P.kAtt2 = 30;
P.lambdaAtt = 8;
P.etaAtt    = 1;
P.psiAtt    = 5;%Dont set to 0

%% ========== CONTROLLER TIMING & FILTERS =================================
P.dtCtrl   = 0.0001;
P.dfilterA = 0;
P.etaDotMax = deg2rad(60);   % 60°/s per Euler axis

%% ========== ACTUATOR LIMITS & constants =============================================
P.tauAct = 0.2; P.kAct = 200;
Fe_max     = 500;
P.FeMin    = 0.05*Fe_max;
P.FeMax    = Fe_max;
P.gimbalMax= 7*pi/180;
end

%% below is a reasonable config incase tuning gets too messy
% function P = rocketConfigEuler()
% %ROCKETCONFIG  All physical constants, gains, trajectory, *and ICs*
% %body info
% R = 0.3; %radius of vehicle
% ISP = 180; %ISP of engine
% %% ========== PHYSICAL CONSTANTS ==========================================
% P.g      = 9.81;
% P.rEng   = [0; 0; -1];
% 
% P.alpha1 = 1/(ISP*9.81);     % mdot coeff. = (ISP*gravity)^(-1) look at appendix B for info
% P.alpha2 = (R^2)/(2*ISP*9.81);     % Jdot coeff. = (R^2 * 1/(2*ISP*gravity)) 
% P.beta   = 0.03;      % throttle → flow-rate
% 
% %% ========== DISTURBANCE MODEL ===========================================
% P.useDist          = false;                 % <-- set TRUE so the lines below take effect
% % --- Force disturbance ---------------------------------------------------
% %here you have the option
% % Body-frame components:  Fx = 20 sin(3 t) , Fy = 0 , Fz = 20 sin(2 t)
% P.distForceGlobal  = true;                % false → you're giving body-frame values
% P.distForceFcn     = @(t) [ 20*sin(3*t) ; ...
%                             0            ; ...
%                             20*sin(2*t) ];
% % --- Moment disturbance --------------------------------------------------
% % Zero roll-, pitch- and yaw-torque
% P.distMomentGlobal = false;                % remains in body frame (anyway zero)
% P.distMomentFcn    = @(t) [ 20 ; 0 ; 0 ];
% 
% %% ================= INITIAL CONDITIONS  ==================================
% % Local variables first (all column vectors)
% pos0   = [0; 0; 0];          % 3×1
% vel0   = [0; 0; 0];          % 3×1
% eul0  =  [0; 0; 0];           % 3×1  (make it a column)
% omega0 = [0; 0; 0];          % 3×1
% 
% % Optional mass / inertia
% m0     = 35;                 % scalar
% J0     = diag([3.5 3.5 0.447]);      % 3×3
% 
% % Build the full 12×1 state vector in one shot
% state0 = [ pos0        ;    % 3
%            vel0        ;    % 3
%            eul0       ;    % 3
%            omega0 ];        % 3
% 
% % --- after building state0 (12×1) ---
% stateDot0 = zeros(size(state0));    % 12×1 zero derivative
% 
% initStruct = struct( ...
%     'pos'   , pos0      , ...
%     'vel'   , vel0      , ...
%     'quat'  , eul0     , ...
%     'omega' , omega0    , ...
%     'mass'  , m0        , ...
%     'J'     , J0        , ...
%     'state' , state0    , ...
%     'stateDot', stateDot0 );   % <— new line
% 
% P.init = initStruct;
% %% ========== NOMINAL 3-D TRAJECTORY ======================================
% P.traj.x0 = 0;   P.traj.xi = 2;   P.traj.xf = 4;
% P.traj.y0 = 0;   P.traj.yi = 2;   P.traj.yf = 4;
% P.traj.z0 = 0;   P.traj.zi = 2;   P.traj.zf = 0;
% P.traj.t1       = 5;
% P.traj.t2       = 5;
% P.traj.dt_blend = 2;
% P.traj.frac_tb  = 0.20;
% % ------------ after your existing traj fields -------------
% P.traj.useStep   = false;         % set true to enable a hard step
% P.traj.stepTime  = 2.0;           % the time (s) at which the jump occurs
% P.traj.stepTarget = [0.1; 0; 0];  % [x; y; z] you want after the step
% % ----------------------------------------------------------
% 
% %% ========== OUTER-LOOP TRANSLATION GAINS ================================
% P.kPos1          = [20  20  6];     % k₁ for x, y, z
% P.kPos2          = [10 10 10];     % k₂ for x, y, z
% P.lambdaPosOuter = [8  8  8];     % λ  for x, y, z
% P.etaPosOuter    = 0.4;         % same η for all axes
% P.psiPosOuter    = [5 5 0.1]; % ψ  per axis
% 
% %% ========== INNER-LOOP (Fz) GAINS =======================================
% P.kPos3 = 10;   P.kPos4 = 10;
% P.lambdaPosInner = 8;
% P.etaPosInner    = 0.5;
% P.psiPosInner    = 5;%dont set to 0, 
% 
% %% ========== ATTITUDE GAINS ==============================================
% P.kAtt1 = 30;  P.kAtt2 = 30;
% P.lambdaAtt = 8;
% P.etaAtt    = 1;
% P.psiAtt    = 5;%Dont set to 0
% 
% %% ========== CONTROLLER TIMING & FILTERS =================================
% P.dtCtrl   = 0.0001;
% P.dfilterA = 0;
% P.etaDotMax = deg2rad(60);   % 60°/s per Euler axis
% 
% %% ========== ACTUATOR LIMITS =============================================
% Fe_max     = 500;
% P.FeMin    = 0.05*Fe_max;
% P.FeMax    = Fe_max;
% P.gimbalMax= 7*pi/180;
% end