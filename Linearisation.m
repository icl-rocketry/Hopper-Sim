clear 
clc
close all

syms x y z U V W p q r phi theta psi real % states
syms xdot ydot zdot Udot Vdot Wdot pdot qdot rdot phidot thetadot psidot real % state derivatives
syms m Ia Izz mdot Iadot Izzdot lx ly lz real % constans
syms T alpha beta real % control inputs

T_etob=[cos(theta)*cos(psi), sin(theta)*sin(phi)*cos(psi)-cos(phi)*sin(psi), cos(phi)*sin(theta)*cos(psi)+sin(phi)*sin(psi);
    cos(theta)*sin(psi), sin(phi)*sin(theta)*sin(psi)+cos(phi)*cos(psi), cos(phi)*sin(theta)*sin(psi)-sin(phi)*cos(psi);
    -sin(theta),sin(phi)*cos(theta), cos(phi)*cos(theta)]';

Xdot=T_etob'*[U;V;W];

AngleCon_btoe=[1 sin(phi)*tan(theta), cos(phi)*tan(theta);
    0, cos(phi), -sin(phi); ...
    0, sin(phi)*sec(theta), cos(phi)*sec(theta)];
EulerAngledot=AngleCon_btoe*[p;q;r];



%% 


