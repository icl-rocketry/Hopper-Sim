clear
clc
clf
close all

syms x y z U V W p q r phi theta psi real % states
syms xdot ydot zdot Udot Vdot Wdot pdot qdot rdot phidot thetadot psidot real % state derivatives
syms m Ia Izz mdot Iadot Izzdot lz g real % constants
syms T alpha beta real % control inputs

x = [x y z phi theta psi U V W p q r]';
xeq = zeros(length(x), 1); % Equilibrium states
u = [T alpha beta]'; % Control input vector
ueq = [m*g 0 0]'; % Equilibrium inputs
h = x;

T_etob=[cos(theta)*cos(psi), sin(theta)*sin(phi)*cos(psi)-cos(phi)*sin(psi), cos(phi)*sin(theta)*cos(psi)+sin(phi)*sin(psi);
    cos(theta)*sin(psi), sin(phi)*sin(theta)*sin(psi)+cos(phi)*cos(psi), cos(phi)*sin(theta)*sin(psi)-sin(phi)*cos(psi);
    -sin(theta),sin(phi)*cos(theta), cos(phi)*cos(theta)]';

Xdot=(T_etob')*[U;V;W];

AngleCon_btoe=[1 sin(phi)*tan(theta), cos(phi)*tan(theta);
    0, cos(phi), -sin(phi); ...
    0, sin(phi)*sec(theta), cos(phi)*sec(theta)];
EulerAngledot=AngleCon_btoe*[p;q;r];

f = [
(m*(V*r-W*q)-mdot*U-m*g*sin(theta)-T*sin(alpha))/m;
(m*(W*p-U*r)-mdot*V+m*g*cos(theta)*sin(phi)+T*cos(alpha)*sin(beta))/m;
(m*(U*q-V*p)-mdot*W+m*g*cos(theta)*cos(phi)-T*cos(alpha)*cos(beta))/m;
(-lz*T*cos(alpha)*sin(beta)-q*r*(Izz-Ia)-Iadot*p)/Ia;
(-lz*T*sin(alpha)-p*r*(Ia-Izz)-Iadot*q)/Ia;
(-Izzdot*r)/Izz;
];

f = [Xdot; EulerAngledot; f];

Axu = jacobian(f,x);
Bxu = jacobian(f,u);
Cxu = jacobian(h,x);
Dxu = jacobian(h,u);

A = subs(Axu,[x; u],[xeq; ueq]);
B = subs(Bxu,[x; u],[xeq; ueq]);
C = subs(Cxu,[x; u],[xeq; ueq]);
D = subs(Dxu,[x; u],[xeq; ueq]);

mval        = 1.4;
mdotval     = 0;
Iaval       = 0.296;
Iadotval    = 0;
Izzval      = 0.05;
Izzdotval   = 0;
lzval       = 0.072;
gval        = 9.81;

csts = [mval; Iaval; Izzval; mdotval; Iadotval; Izzdotval; lzval; gval];

A = double(subs(A, [m; Ia; Izz; mdot; Iadot; Izzdot; lz; g], csts));
B = double(subs(B, [m; Ia; Izz; mdot; Iadot; Izzdot; lz; g], csts));
C = double(subs(C, [m; Ia; Izz; mdot; Iadot; Izzdot; lz; g], csts));
D = double(subs(D, [m; Ia; Izz; mdot; Iadot; Izzdot; lz; g], csts));

Ared = A([1:5,7:end-1], [1:5,7:end-1]);
Bred = B([1:5,7:end-1], :);

[reachable,observable] = isReachOrObsv(Ared,Bred,eye(10));

Aaug = [Ared, zeros(10,3); eye(3), zeros(3,7), zeros(3)];
Baug = [Bred; zeros(3)];

Ts = 0.005;

Q = eye(13);

R = diag(ones(1,3));

[Kd, S, P] = lqrd(Aaug, Baug, Q, R, Ts);

K = Kd(:,1:10);
Ki = Kd(:,11:13);

params = [mval; mdotval; Iaval; Iadotval; Izzval; Izzdotval; lzval];

Cred = [eye(3), zeros(3,7)];
Kf = -(Cred*((Ared-Bred*K)\Bred))\eye(3);

Teq = double(subs(ueq(1), [m; g], [mval; gval]));
Tmax = 20;
gimble_max = 20;

out = sim('Hopper_lqr', 'StartTime', '0', 'StopTime', '60', 'FixedStep', num2str(Ts));

figure
ylabs = {'x', 'y', 'z'};
for i = 1:3
    subplot(1,3,i)
    plot(out.states.time, squeeze(out.states.signals.values(i,1,:)))
    hold on; grid on; box on
    plot(out.refs.time, out.refs.signals.values(:,i))
    xlabel('t'); ylabel(ylabs{i})
end

figure
ylabs = {'phi', 'theta', 'psi', 'U', 'V', 'W', 'p', 'q', 'r'};
for i = 1:9
    subplot(3,3,i)
    plot(out.states.time, squeeze(out.states.signals.values(3+i,1,:)))
    xlabel('t'); ylabel(ylabs{i})
    grid on; box on
end

figure
ylabs = {'T', 'alpha', 'beta'};
ylims = {[0 Tmax], deg2rad([-5 5]), deg2rad([-5 5])};
for i = 1:3
    subplot(1,3,i)
    plot(out.controls.time, squeeze(out.controls.signals.values(i,1,:)))
    grid on; box on
    xlabel('t'); ylabel(ylabs{i}); ylim(ylims{i})
end
