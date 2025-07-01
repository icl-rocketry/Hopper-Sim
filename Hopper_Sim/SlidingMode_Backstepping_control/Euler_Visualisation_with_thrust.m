%% run_and_plot.m
% 1) Build your P struct and put it in the base workspace
P = rocketConfigEuler();
assignin('base','P',P);

% 2) Load & simulate the model, with StructureWithTime output
model = 'sixDOFTrial_Euler';
load_system(model);
simOut = sim(model,...
    'StopTime'       ,'10',...
    'SaveFormat'     ,'StructureWithTime',...   % timeseries output
    'TimeSaveName'   ,'simTime',...
    'OutputSaveName' ,'simOut');               % top‐level struct

% 3) Grab the logged signals directly from simOut
t_ts   = simOut.t;    % timeseries for simulation time
s_ts   = simOut.s;    % timeseries for full 12×1 state
Fx_ts  = simOut.Fx;   % timeseries for Fx
Fy_ts  = simOut.Fy;   % timeseries for Fy
Fz_ts  = simOut.Fz;   % timeseries for Fz

% optional: reference trajectories if you logged them
hasRef = isfield(simOut,'x_ref') && isfield(simOut,'y_ref') && isfield(simOut,'z_ref');
if hasRef
  xref_ts = simOut.x_ref;
  yref_ts = simOut.y_ref;
  zref_ts = simOut.z_ref;
end

% 4) Extract raw data arrays
t_data  =   t_ts.Time;           % N×1 vector of time
raw_s   =   squeeze(s_ts.Data);  % might be N×12 or 12×N
Fx_data =   Fx_ts.Data(:);       % N×1 
Fy_data =   Fy_ts.Data(:);
Fz_data =   Fz_ts.Data(:);

% fix orientation of the state array
if size(raw_s,2)==12
  s_data = raw_s;      % already N×12
elseif size(raw_s,1)==12
  s_data = raw_s.';    % transpose to N×12
else
  error('Unexpected s_ts.Data size %s', mat2str(size(raw_s)));
end

% extract references
if hasRef
  x_ref_data = xref_ts.Data(:);
  y_ref_data = yref_ts.Data(:);
  z_ref_data = zref_ts.Data(:);
else
  x_ref_data = []; y_ref_data = []; z_ref_data = [];
end

% harmonize lengths
lenList = [ numel(t_data), size(s_data,1), ...
            numel(Fx_data), numel(Fy_data), numel(Fz_data) ];
if hasRef
  lenList = [ lenList, numel(x_ref_data), numel(y_ref_data), numel(z_ref_data) ];
end
N = min(lenList);
t_data  = t_data(1:N);
s_data  = s_data(1:N,:);
Fx_data = Fx_data(1:N);
Fy_data = Fy_data(1:N);
Fz_data = Fz_data(1:N);
if hasRef
  x_ref_data = x_ref_data(1:N);
  y_ref_data = y_ref_data(1:N);
  z_ref_data = z_ref_data(1:N);
end

% 5) Split state vector
x_data   = s_data(:,1);
y_data   = s_data(:,2);
z_data   = s_data(:,3);
eul_data = s_data(:,7:9);   % [phi theta psi] in ZYX order

% 6) Prepare animation
scaleFactor     = 2;
showCOM         = false;
showThrust      = true;
skipFactor      = 1;
L               = 1;          % rocket length [m]
maxThrust       = 500;        % expected peak |F| [N]
maxThrustLength = 0.5;        % plume length at max thrust
bodyVec_b       = [0 0 L/2]'; % nose direction in body frame

figure('Name','Rocket Animation (Euler)','Color','w');
set(gcf, ...
  'DefaultAxesFontSize',get(groot,'defaultAxesFontSize')*scaleFactor, ...
  'DefaultTextFontSize',get(groot,'defaultTextFontSize')*scaleFactor, ...
  'DefaultLegendFontSize',get(groot,'defaultLegendFontSize')*scaleFactor);
hold on; grid on; axis equal;
axis([-0.5 4.5  -1 4.5  -0.5 3]);
view(35,25); box on;

% plot static reference path
if hasRef
  plot3(x_ref_data,y_ref_data,z_ref_data,'g--','LineWidth',2,'DisplayName','Ref Traj');
end

% placeholders
hBody   = plot3(NaN,NaN,NaN,'b-','LineWidth',4,'DisplayName','Rocket Body');
hCOM    = []; hThrust = [];
if showCOM
  hCOM = plot3(NaN,NaN,NaN,'ro','MarkerFaceColor','r','DisplayName','CoM');
end
hPath   = plot3(NaN,NaN,NaN,'k--','DisplayName','Actual Path');
if showThrust
  hThrust = plot3(NaN,NaN,NaN,'r-','LineWidth',3,'DisplayName','Thrust');
end

xlabel('X'); ylabel('Y'); zlabel('Z');
title('Rocket Animation with Thrust');

% 7) Main loop
for i=1:skipFactor:N
  % current pose
  pos = [x_data(i); y_data(i); z_data(i)];
  R   = eul2rotm(eul_data(i,:));  % returns 3×3 from MATLAB

  nose = pos + R*bodyVec_b;
  tail = pos - R*bodyVec_b;
  set(hBody,'XData',[tail(1) nose(1)],...
            'YData',[tail(2) nose(2)],...
            'ZData',[tail(3) nose(3)]);

  if ~isempty(hCOM)
    set(hCOM,'XData',pos(1),'YData',pos(2),'ZData',pos(3));
  end

  % trajectory so far
  set(hPath,'XData',x_data(1:i),...
            'YData',y_data(1:i),...
            'ZData',z_data(1:i));

  % thrust vector
  if showThrust
    tb = [Fx_data(i); Fy_data(i); Fz_data(i)];
    mag = norm(tb);
    if mag>0
      tv = (tb/mag)* (mag/maxThrust)*maxThrustLength;
      tv_w = R*tv;
      set(hThrust,'XData',[tail(1) tail(1)-tv_w(1)],...
                  'YData',[tail(2) tail(2)-tv_w(2)],...
                  'ZData',[tail(3) tail(3)-tv_w(3)],...
                  'Visible','on');
    else
      set(hThrust,'Visible','off');
    end
  end

  title(sprintf('Time = %.2f s', t_data(i)));
  drawnow limitrate;
end