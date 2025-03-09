clear
clc
clf
close all

% Define symbolic variables
syms x y z U V W p q r phi theta psi real % states
syms xdot ydot zdot Udot Vdot Wdot pdot qdot rdot phidot thetadot psidot real % state derivatives
syms m Ia Izz mdot Iadot Izzdot lz g real % constants
syms T alpha beta real % control inputs

x = [x y z phi theta psi U V W p q r]'; % State vector
xeq = zeros(length(x), 1); % Equilibrium states
u = [T alpha beta]'; % Control input vector
ueq = [m*g 0 0]'; % Equilibrium inputs
h = x; % Assume perfect state feedback

% Earth to body vector transformation matrix using Euler angles
T_etob=[cos(theta)*cos(psi), sin(theta)*sin(phi)*cos(psi)-cos(phi)*sin(psi), cos(phi)*sin(theta)*cos(psi)+sin(phi)*sin(psi);
    cos(theta)*sin(psi), sin(phi)*sin(theta)*sin(psi)+cos(phi)*cos(psi), cos(phi)*sin(theta)*sin(psi)-sin(phi)*cos(psi);
    -sin(theta),sin(phi)*cos(theta), cos(phi)*cos(theta)]';

% Compute velocity in inertial frame
Xdot=(T_etob')*[U;V;W];

% Compute rates of change of Euler angles from body rotation rates
AngleCon_btoe=[1 sin(phi)*tan(theta), cos(phi)*tan(theta);
    0, cos(phi), -sin(phi); ...
    0, sin(phi)*sec(theta), cos(phi)*sec(theta)];
EulerAngledot=AngleCon_btoe*[p;q;r];

% Assemble state derivative function
f = [
(m*(V*r-W*q)-mdot*U-m*g*sin(theta)-T*sin(alpha))/m;
(m*(W*p-U*r)-mdot*V+m*g*cos(theta)*sin(phi)+T*cos(alpha)*sin(beta))/m;
(m*(U*q-V*p)-mdot*W+m*g*cos(theta)*cos(phi)-T*cos(alpha)*cos(beta))/m;
(-lz*T*cos(alpha)*sin(beta)-q*r*(Izz-Ia)-Iadot*p)/Ia;
(-lz*T*sin(alpha)-p*r*(Ia-Izz)-Iadot*q)/Ia;
(-Izzdot*r)/Izz;
];
f = [Xdot; EulerAngledot; f];

% Compute symbolic Jacobians
Axu = jacobian(f,x);
Bxu = jacobian(f,u);
Cxu = jacobian(h,x);
Dxu = jacobian(h,u);

% Substitute equilibrium states and controls
A = subs(Axu,[x; u],[xeq; ueq]);
B = subs(Bxu,[x; u],[xeq; ueq]);
C = subs(Cxu,[x; u],[xeq; ueq]);
D = subs(Dxu,[x; u],[xeq; ueq]);

% Define plant parameters
mval        = 1.4;
mdotval     = 0;
Iaval       = 0.296;
Iadotval    = 0;
Izzval      = 0.05;
Izzdotval   = 0;
lzval       = 0.072;
gval        = 9.81;

csts = [mval; Iaval; Izzval; mdotval; Iadotval; Izzdotval; lzval; gval];

% Convert symbolic matrices to double-precision
A = double(subs(A, [m; Ia; Izz; mdot; Iadot; Izzdot; lz; g], csts));
B = double(subs(B, [m; Ia; Izz; mdot; Iadot; Izzdot; lz; g], csts));
C = double(subs(C, [m; Ia; Izz; mdot; Iadot; Izzdot; lz; g], csts));
D = double(subs(D, [m; Ia; Izz; mdot; Iadot; Izzdot; lz; g], csts));

% Reduce matrices to exclude yaw and yaw rate for reachability
Ared = A([1:5,7:end-1], [1:5,7:end-1]);
Bred = B([1:5,7:end-1], :);

% Check reachability and observability
[reachable,observable] = isReachOrObsv(Ared,Bred,eye(10));

% Augment matrices for position integral action
Aaug = [Ared, zeros(10,3); eye(3), zeros(3,7), zeros(3)];
Baug = [Bred; zeros(3)];
Ts = 0.005; % Sample time

% LQR gains
Q = diag([1 1 16,...
          3 3,...
          1 1 8, ...
          0.5 0.5,...
          0.04 0.04 0.1]);
R = diag([1 1.2 1.2]);
eta = [0.6, 0.6, 10];

% Compute discrete time LQR solution
[Kd, S, P] = lqrd(Aaug, Baug, Q, R, Ts);

% Separate gains into state gains and integral gains
K = Kd(:,1:10);
Ki = Kd(:,11:13);

params = [mval; mdotval; Iaval; Iadotval; Izzval; Izzdotval; lzval];

% Compute feedforward gain using optimal control law
Cred = [eye(3), zeros(3,7)];
Kf = -(Cred*((Ared-Bred*K)\Bred))\eye(3);

% Define control parameters for Simulink
Teq = double(subs(ueq(1), [m; g], [mval; gval]));

% Run simulation
out = sim('Hopper_lqr', 'StartTime', '0', 'StopTime', '60', 'FixedStep', num2str(Ts));

%% Plot results
figure
ylabs = {'x', 'y', 'z'};
for i = 1:3
    subplot(1,3,i)
    plot(out.states.time, out.states.signals.values(:,i))
    hold on; grid on; box on
    plot(out.refs.time, out.refs.signals.values(:,i))
    xlabel('t'); ylabel(ylabs{i})
end

figure
ylabs = {'phi', 'theta', 'psi', 'U', 'V', 'W', 'p', 'q', 'r'};
for i = 1:9
    subplot(3,3,i)
    plot(out.states.time, out.states.signals.values(:,3+i))
    xlabel('t'); ylabel(ylabs{i})
    grid on; box on
end

figure
ylabs = {'T', 'alpha', 'beta'};
%ylims = {[0 Tmax], deg2rad([-5 5]), deg2rad([-5 5])};
for i = 1:3
    subplot(1,3,i)
    plot(out.controls.time, out.controls.signals.values(:,i))
    grid on; box on
    xlabel('t'); ylabel(ylabs{i}); %ylim(ylims{i})
end

%%

t = out.states.time;
x = out.states.signals.values(:,1);
y = out.states.signals.values(:,2);
z = out.states.signals.values(:,3);
phi = out.states.signals.values(:,4);
theta = out.states.signals.values(:,5);
psi = out.states.signals.values(:,6);
thrust = out.controls.signals.values(:,1);
alpha = out.controls.signals.values(:,2);
beta = out.controls.signals.values(:,3);


euler_angles_array=[phi, theta, psi];
position_earth_array=[x, y, z];
time_array=t;
thrust_array = [thrust, alpha, beta];

Length = [0.2 0.2 0.6]; % Side length of the cube
figure;
axis equal;
xlim([-5 5])
ylim([-5 5])
zlim([0 20])
xlabel('X');
ylabel('Y');
zlabel('Z');
title('3D Wireframe Cube');
grid on;
view(3);

% Loop through each time step
for i = 1:length(time_array)
    cg = position_earth_array(i, 1:3);
    cg(3) = -cg(3); % Adjust z-coordinate
    cg(2) = -cg(2); % Adjust y-coordinate
    phi=euler_angles_array(i,1);
    theta=euler_angles_array(i,2);
    psi=euler_angles_array(i,3);

    T_etob=[cos(theta)*cos(psi), sin(theta)*sin(phi)*cos(psi)-cos(phi)*sin(psi), cos(phi)*sin(theta)*cos(psi)+sin(phi)*sin(psi);
    cos(theta)*sin(psi), sin(phi)*sin(theta)*sin(psi)+cos(phi)*cos(psi), cos(phi)*sin(theta)*cos(psi)-sin(phi)*cos(psi);
    -sin(theta),sin(phi)*cos(theta), cos(phi)*cos(theta)]';

    cla; % Clear current axes
    plotRocket3D(cg, Length,T_etob,thrust_array(i,:));
    pause(0.001);
end

function plotRocket3D(cg, lengths, Tetob,thrust_array)
    % lengths is a vector [length, width, height]
    length = lengths(1);
    width = lengths(2);
    height = lengths(3);

    % Define the half side lengths
    halfLength = length / 2;
    halfWidth = width / 2;
    halfHeight = height / 2;

    % Define the vertices of the cuboid
    vertices = [
        -halfLength, -halfWidth, -halfHeight;
        halfLength, -halfWidth, -halfHeight;
        halfLength, halfWidth, -halfHeight;
        -halfLength, halfWidth, -halfHeight;
        -halfLength, -halfWidth, halfHeight;
        halfLength, -halfWidth, halfHeight;
        halfLength, halfWidth, halfHeight;
        -halfLength, halfWidth, halfHeight;
    ];

    % Transform vertices using Tetob matrix
    [row, ~] = size(vertices);
    newVertices = zeros(size(vertices));
    for v = 1:row
        newVertices(v, :) = (Tetob * vertices(v, :)')';
    end

    % Shift vertices to be centered at cg
    newVertices = newVertices + cg;

    % Define the edges of the cuboid
    edges = [
        1, 2; 2, 3; 3, 4; 4, 1; % bottom edges
        5, 6; 6, 7; 7, 8; 8, 5; % top edges
        1, 5; 2, 6; 3, 7; 4, 8; % vertical edges
    ];

    nose=Tetob*[0;0;halfHeight+0.5]+cg';

    % Plot body
    hold on;
    for i = 1:size(edges, 1)
        plot3(newVertices(edges(i, :), 1), newVertices(edges(i, :), 2), newVertices(edges(i, :), 3), 'b');
    end

    %plot nose
    for j=5:row
    plot3([nose(1);newVertices(j,1)],[nose(2);newVertices(j,2)],[nose(3);newVertices(j,3)],'r');
    end 
  
    
   support_end=vertices(1:4,:)+[-0.2 -0.2 -0.2; 0.2 -0.2 -0.2; 0.2 0.2 -0.2;-0.2 0.2 -0.2];
    
    %rotate support coordinates
    for k=1:4
        newSupport_end(k,:)=(Tetob * support_end(k, :)')';
    end 

    newSupport_end=newSupport_end+cg;

    %plot landing supports
     for L=1:4
        plot3([newVertices(L,1);newSupport_end(L,1)],[newVertices(L,2);newSupport_end(L,2)],[newVertices(L,3);newSupport_end(L,3)],'b');
    end

    %plot thrust

    thrust_start=Tetob*[0;0;-halfHeight]+cg';
    
    
    thrust_scaling=0.0005;
    alpha=thrust_array(1);
    beta=thrust_array(2);
    T=thrust_array(3);
    Tx=-T*sin(alpha)*  thrust_scaling;
    Ty=-T*cos(alpha)*sin(beta)*  thrust_scaling;
    Tz=-T*cos(alpha)*cos(beta)*  thrust_scaling;

    thrust_end=Tetob*[Tx;Ty;-halfHeight+Tz]+cg';

   
    plot3([thrust_start(1);thrust_end(1)],[thrust_start(2);thrust_end(2)],[thrust_start(3);thrust_end(3)],'g')
          hold off;
    
end