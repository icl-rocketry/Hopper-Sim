clear 
clc
close all

syms x y z U V W p q r phi theta psi real % states
syms xdot ydot zdot Udot Vdot Wdot pdot qdot rdot phidot thetadot psidot real % state derivatives
syms m Ia Izz mdot Iadot Izzdot lx ly lz real % constans
syms T alpha beta real % control inputs
gb=9.81;

T_etob=[cos(theta)*cos(psi), sin(theta)*sin(phi)*cos(psi)-cos(phi)*sin(psi), cos(phi)*sin(theta)*cos(psi)+sin(phi)*sin(psi);
    cos(theta)*sin(psi), sin(phi)*sin(theta)*sin(psi)+cos(phi)*cos(psi), cos(phi)*sin(theta)*sin(psi)-sin(phi)*cos(psi);
    -sin(theta),sin(phi)*cos(theta), cos(phi)*cos(theta)]';

Xdot=T_etob'*[U;V;W];

AngleCon_btoe=[1 sin(phi)*tan(theta), cos(phi)*tan(theta);
    0, cos(phi), -sin(phi); ...
    0, sin(phi)*sec(theta), cos(phi)*sec(theta)];
EulerAngledot=AngleCon_btoe*[p;q;r];

Tx=-T*sin(alpha);
Ty=T*cos(alpha)*sin(beta); %this should be positive,  but accounted in the gains 
Tz=-T*cos(alpha)*cos(beta);

ge=T_etob*[0;0;gb];
gx=ge(1);
gy=ge(2);
gz=ge(3);

% lx=thrust_pos(1)-cg(1);
% ly=thrust_pos(2)-cg(2);
% lz=thrust_pos(3)-cg(3);

Fx=Tx+m*gx;
Fy=Ty+m*gy;
Fz=Tz+m*gz;

Moments=cross([lx;ly;lz],[Tx;Ty;Tz]);
L=Moments(1);
M=Moments(2);
N=Moments(3);

Udot=Fx/m-(W*q-V*r)-mdot*U/m;
Vdot=Fy/m-(U*r-p*W)-mdot*V/m;
Wdot=Fz/m-(V*p-U*q)-mdot*W/m;

pdot=(L-q*r*(Izz-Ia)-Iadot*p)/Ia;
qdot=(M-r*p*(Ia-Izz)-Iadot*q)/Ia;
rdot=(N-p*q*(Ia-Ia)-Izzdot*r)/Izz;

Xddot=T_etob'*[Udot;Vdot;Wdot];

f=[Xdot;Udot;Vdot;Wdot;EulerAngledot;pdot;qdot;rdot;Xddot];
states=[x y z U V W phi theta psi p q r xdot ydot zdot ];
control=[T alpha beta];

% paramValues=[0.296 0.05 0 0 72e-3 1.4 0 0 0];
% f = subs(f, [Ia Izz lx ly lz m mdot Iadot Izzdot], paramValues);      
% f = simplify(f);

% A = jacobian(f,states);
% B = jacobian(f,control);
% 
% f_xeq = subs(f, states, zeros(1,15)) ;                  
% [T_input, alpha_input, beta_input] = solve(f_xeq == 0, control);     
% input_Eq = double([T_input alpha_input beta_input]);  
% 
% Aeq=subs(A,[states control],[zeros(1,15) input_Eq]);
% Beq=subs(B,[states control],[zeros(1,15) input_Eq]);
%% 


syms Kp_theta Ki_theta Kd_theta Kp_x Ki_x Kd_x z_x z_theta z_xdot z_thetadot rx real
syms Kp_phi Ki_phi Kd_phi Kp_y Ki_y Kd_y z_y z_phi z_ydot z_phidot ry real
syms  Kp_z Ki_z Kd_z z_z  rz  z_zdot real



f_linear=Aeq*states'+Beq*control';
xdot=f_linear(1);

xddot=f_linear(end-1);
yddot=f_linear(end-2);
ydot=f_linear(2);
thetadot=f_linear(8);
phidot=f_linear(7);
alpha_expr=Kp_theta*Kp_x*(rx-x)+Kp_theta*Ki_x*z_x-Kp_theta*Kd_x*xdot-Kp_theta*theta+Ki_theta*z_theta-Kd_theta*Kp_x*xdot+Kd_theta*Ki_x*(rx-x)-Kd_theta*Kd_x*xddot-Kd_theta*thetadot;
beta_expr = Kp_phi * Kp_y * (ry - y) ...
        + Kp_phi * Ki_y * z_y ...
        - Kp_phi * Kd_y * ydot ...
        - Kp_phi * phi ...
        + Ki_phi * z_phi ...
        - Kd_phi * Kp_y * ydot ...
        + Kd_phi * Ki_y * (ry - y) ...
        - Kd_phi * Kd_y * yddot ...
        - Kd_phi * phidot;
zdot=f_linear(3);
T=Kp_z*(rz-z)+Ki_z*z_z-Kd_z*zdot;

% Define symbolic equations for alpha and beta
eq1 = alpha == alpha_expr;
eq2 = beta == beta_expr;

% Rearrange the equations to isolate alpha and beta
[alpha_sol, beta_sol] = solve([eq1, eq2], [alpha, beta]);

states_new = [states z_x z_y z_phi z_theta];
r = [rx; ry];

% Express u = Cx + Dr
C_alpha = simplify(jacobian(alpha_sol, states_new));
C_beta = simplify(jacobian(beta_sol, states_new));
D_alpha = simplify(jacobian(alpha_sol, r));
D_beta = simplify(jacobian(beta_sol, r));

% Combine into matrix forms
C = [C_alpha; C_beta];
D = [D_alpha; D_beta];

states_cl=[states z_x z_y z_z z_phi z_theta ];




%% 
