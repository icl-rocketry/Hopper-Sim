%% plot_6dof_errors.m
% Plots each of the 6 outer-loop tracking errors alongside its matched disturbance.

%--- 0) Pull data out of 'out' struct ------------------------------------
t_data     = out.t.Time;                    % N×1
S          = squeeze(out.s.Data);           % N×12 or 12×N
E_ref_full = squeeze(out.Euler_d.Data);     % N×3 or 3×N
D_full     = squeeze(out.dist.Data);        % N×6 or 6×N

%--- 1) Fix orientations if necessary -----------------------------------
if size(S,2)~=12,    S = S.';       end
if size(E_ref_full,2)~=3, E_ref_full = E_ref_full.'; end
if size(D_full,2)~=6,  D_full = D_full.';  end

%--- 2) Extract states & references -------------------------------------
x_data     = S(:,1);  y_data = S(:,2);  z_data = S(:,3);
phi_data   = S(:,7);  theta_data = S(:,8);  psi_data = S(:,9);

phr        = E_ref_full(:,1);   % φ_ref
thr        = E_ref_full(:,2);   % θ_ref
psr        = E_ref_full(:,3);   % ψ_ref

xr         = out.x_ref.Data(:);
yr         = out.y_ref.Data(:);
zr         = out.z_ref.Data(:);

Fx_dist    = D_full(:,1);
Fy_dist    = D_full(:,2);
Fz_dist    = D_full(:,3);
Mx_dist    = D_full(:,4);
My_dist    = D_full(:,5);
Mz_dist    = D_full(:,6);

%--- 3) Trim to common length -------------------------------------------
L = min([ numel(t_data), numel(x_data), numel(phr), numel(Fx_dist) ]);
t_data     = t_data    (1:L);
x_data     = x_data    (1:L);
y_data     = y_data    (1:L);
z_data     = z_data    (1:L);
phi_data   = phi_data  (1:L);
theta_data = theta_data(1:L);
psi_data   = psi_data  (1:L);

xr         = xr        (1:L);
yr         = yr        (1:L);
zr         = zr        (1:L);

phr        = phr       (1:L);
thr        = thr       (1:L);
psr        = psr       (1:L);

Fx_dist    = Fx_dist   (1:L);
Fy_dist    = Fy_dist   (1:L);
Fz_dist    = Fz_dist   (1:L);
Mx_dist    = Mx_dist   (1:L);
My_dist    = My_dist   (1:L);
Mz_dist    = Mz_dist   (1:L);

%--- 4) Compute errors ---------------------------------------------------
ex     = x_data - xr;
ey     = y_data - yr;
ez     = z_data - zr;
ephi   = wrapToPi(phi_data   - phr);
etheta = wrapToPi(theta_data - thr);
epsi   = wrapToPi(psi_data   - psr);

%--- 5) Pack for loop ----------------------------------------------------
states   = { x_data, y_data, z_data, phi_data,   theta_data,   psi_data };
refs     = { xr,      yr,      zr,      phr,        thr,         psr        };
errors   = { ex,      ey,      ez,      ephi,       etheta,      epsi       };
disturbs = { Fx_dist, Fy_dist, Fz_dist, Mx_dist,    My_dist,     Mz_dist    };
ylabs    = {'X [m]','Y [m]','Z [m]','\phi [rad]','\theta [rad]','\psi [rad]'};
distlabs = {'F_x [N]','F_y [N]','F_z [N]','M_x [Nm]','M_y [Nm]','M_z [Nm]'};
titles   = {'(a)','(b)','(c)','(d)','(e)','(f)'};

%--- 6) Plot translations (X,Y,Z) ----------------------------------------
fig1 = figure('Name','Translational Errors & Disturbances','Color','w', ...
              'Units','normalized','Position',[.1 .1 .4 .8]);
fontsize = 12;
legendfontsize = 8;
for k = 1:3                      % X, Y, Z
  ax = subplot(4,1,k); hold(ax,'on');

  % left axis: measurement vs reference
  yyaxis(ax,'left');
    p1 = plot(t_data, states{k}, 'b-','LineWidth',1.3);
    p2 = plot(t_data, refs  {k}, 'r--','LineWidth',1.3);
    ylabel(ylabs{k},'FontSize',fontsize);
    grid on;

  % right axis: tracking error
  yyaxis(ax,'right');
    p3 = plot(t_data, errors{k}, 'Color',[0 .5 0],'LineWidth',1.3);
    ax.YAxis(2).Color = [0 .5 0];                % make RHS axis green
    ylabel([ylabs{k} ' error'],'FontSize',fontsize);
    M = max(abs(errors{k}));
    ylim([-1.1*M,1.1*M]);

  % combined legend
  legend(ax, [p1 p2 p3], ...
         {'meas','ref',[ylabs{k} ' error']}, ...
         'Location','best','FontSize',legendfontsize);

  title(sprintf('%s  %s', titles{k}, ylabs{k}),'FontSize',fontsize);
  if k<3, set(ax,'XTickLabel',''); end
end

% 4th row: all force-disturbances together
ax = subplot(4,1,4); hold(ax,'on');
plot(t_data, Fx_dist,'r-','LineWidth',1.3);
plot(t_data, Fy_dist,'b-','LineWidth',1.3);
plot(t_data, Fz_dist,'k-','LineWidth',1.3);
xlabel('Time [s]','FontSize',fontsize);
ylabel('Force Disturbances [N]','FontSize',fontsize);
legend('F_x','F_y','F_z','Location','best','FontSize',legendfontsize);
grid on;

sgtitle('Translational Outer-Loop: Errors & Force Disturbances','FontSize',16);



%--- 7) Plot rotations (φ,θ,ψ) ------------------------------------------
fig2 = figure('Name','Angular Errors & Disturbances','Color','w', ...
              'Units','normalized','Position',[.55 .1 .4 .8]);
for k = 4:6                      % φ, θ, ψ
  ax = subplot(4,1,k-3); hold(ax,'on');

  % left axis: measurement vs reference
  yyaxis(ax,'left');
    p1 = plot(t_data, states{k}, 'b-','LineWidth',1.3);
    p2 = plot(t_data, refs  {k}, 'r--','LineWidth',1.3);
    ylabel(ylabs{k},'FontSize',fontsize);
    grid on;

  % right axis: tracking error
  yyaxis(ax,'right');
    p3 = plot(t_data, errors{k}, 'Color',[0 .5 0],'LineWidth',1.3);
    ax.YAxis(2).Color = [0 .5 0];                % green RHS axis
    ylabel([ylabs{k} ' error'],'FontSize',fontsize);
    M = max(abs(errors{k}));
    ylim([-1.1*M,1.1*M]);

  % combined legend
  legend(ax, [p1 p2 p3], ...
         {'meas','ref',[ylabs{k} ' error']}, ...
         'Location','best','FontSize',legendfontsize);

  title(sprintf('%s  %s', titles{k}, ylabs{k}),'FontSize',fontsize);
  if k<6, set(ax,'XTickLabel',''); end
end

% 4th row: all moment disturbances together
ax = subplot(4,1,4); hold(ax,'on');
plot(t_data, Mx_dist,'r-','LineWidth',1.3);
plot(t_data, My_dist,'b-','LineWidth',1.3);
plot(t_data, Mz_dist,'k-','LineWidth',1.3);
xlabel('Time [s]','FontSize',fontsize);
ylabel('Moment Disturbances [Nm]','FontSize',fontsize);
legend('M_x','M_y','M_z','Location','best','FontSize',legendfontsize);
grid on;

sgtitle('Angular Outer-Loop: Errors & Moment Disturbances','FontSize',16);