% out = sim('HopperPlant_Current.slx','StopTime', '60');
% euler_angles=get(out,"euler_angles");
% position_earth=get(out,'position');
% thrust=get(out,'thrust');
% error=get(out,'error');
% save("hopper_sim","euler_angles","position_earth","thrust")
% time_array=euler_angles.time;
% euler_angles_array=euler_angles.data;
% position_earth_array=position_earth.data;
% thrust_array=thrust.data;
% error_array=error.data;

clear
clc
clf
close all
global counter
counter=1;
height_setpoint=50;
max_thrust=500;
y_setpoint=2;
x_setpoint=0;
% Kp = 20;
% Ki = 1.5;
% Kd = 70;
% RMSE = cost_func(Kp, Ki, Kd);

objective = @(K) cost_func(K(1), K(2), K(3),K(4), K(5), K(6),K(7), K(8), K(9),K(10), K(11), K(12),K(13), K(14), K(15));
lb = [0, 0, 20,ones(1,12)*-20];
ub = [100 20 200,ones(1,12)*0];


param = particleswarm(objective, 15, lb, ub);


%% parameters


% assignin("base","Kp_z",Kp_z);
% assignin("base","Ki_z",Ki_z);
% assignin("base","Kd_z",Kd_z);

% y controller 
% assignin("base","Kp_y",Kp_y);
% assignin("base","Ki_y",Ki_y);
% assignin("base","Kd_y",Kd_y);
% assignin("base","y_setpoint",y_setpoint);

function RMSE = cost_func(Kp, Ki, Kd,Kp_x, Ki_x, Kd_x,Kp_y, Ki_y, Kd_y,Kp_alpha, Ki_alpha, Kd_alpha,Kp_beta, Ki_beta, Kd_beta)
   
    global counter
    assignin('base', 'Kp_z', Kp);
    assignin('base', 'Ki_z', Ki);
    assignin('base', 'Kd_z', Kd);

    assignin('base', 'Kp_x', Kp_x);
    assignin('base', 'Ki_x', Ki_x);
    assignin('base', 'Kd_x', Kd_x);

     assignin('base', 'Kp_y', Kp_y);
    assignin('base', 'Ki_y', Ki_y);
    assignin('base', 'Kd_y', Kd_y);


    assignin('base', 'Kp_alpha', Kp_alpha);
    assignin('base', 'Ki_alpha', Ki_alpha);
    assignin('base', 'Kd_alpha', Kd_alpha);


    assignin('base', 'Kp_beta', Kp_beta);
    assignin('base', 'Ki_beta', Ki_beta);
    assignin('base', 'Kd_beta', Kd_beta);


    %simOut = sim('HopperPlant_Current.slx', 'StopTime', '60', 'SaveOutput', 'on');

    out = sim('HopperPlant_Current_optimisation.slx', 0:0.05:60);

    RMSE = sqrt(mean(out.error.data(:,1).^2+out.error.data(:,2).^2+out.error.data(:,3).^2));

    % sys_out = % GET signal
    % ref_out = % GET reference
    % 
    % RMSE = sqrt(mean((sys_out - ref_out).^2));

    disp(['Kp: ' num2str(Kp), ' , Ki: ', num2str(Ki), ' , Kd: ', num2str(Kd), ' , RMSE: ', num2str(RMSE)]);
    disp(['x:','Kp: ' num2str(Kp_x), ' , Ki: ', num2str(Ki_x), ' , Kd: ', num2str(Kd_x), ' , RMSE: ', num2str(RMSE)]);
    disp(['y:','Kp: ' num2str(Kp_y), ' , Ki: ', num2str(Ki_y), ' , Kd: ', num2str(Kd_y), ' , RMSE: ', num2str(RMSE)]);
    disp(['alpha','Kp: ' num2str(Kp_alpha), ' , Ki: ', num2str(Ki_alpha), ' , Kd: ', num2str(Kd_alpha), ' , RMSE: ', num2str(RMSE)]);
    disp(['beta','Kp: ' num2str(Kp_beta), ' , Ki: ', num2str(Ki_beta), ' , Kd: ', num2str(Kd_beta), ' , RMSE: ', num2str(RMSE)]);
    
    
    if RMSE<1
    
        goodGains(counter,:)=[Kp, Ki, Kd,Kp_x, Ki_x, Kd_x,Kp_y, Ki_y, Kd_y,Kp_alpha, Ki_alpha, Kd_alpha,Kp_beta, Ki_beta, Kd_beta];
        counter=counter+1;
    end 

    disp(counter)
end
%% 
Kp_z=94.5595 ;
ki_z=0.078559;
Kd_z=78.8685;



    assignin('base', 'Kp_x', 1);
    assignin('base', 'Ki_x', 1);
    assignin('base', 'Kd_x', 1);

     assignin('base', 'Kp_y', 1);
    assignin('base', 'Ki_y', 1);
    assignin('base', 'Kd_y', 1);


    assignin('base', 'Kp_alpha', 1);
    assignin('base', 'Ki_alpha', 1);
    assignin('base', 'Kd_alpha', 1);


    assignin('base', 'Kp_beta', 1);
    assignin('base', 'Ki_beta', 1);
    assignin('base', 'Kd_beta', 1);
height_setpoint=50;
max_thrust=500;
y_setpoint=2;
x_setpoint=0;