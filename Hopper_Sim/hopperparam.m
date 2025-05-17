hopper.airframemass=1.4;
hopper.fuelmass=0.0;

hopper.Ix=units(0.296,'m^4','m^4');
hopper.Iy=units(0.296,'m^4','m^4');
hopper.Iz=units(0.05,'m^4','m^4');

hopper.cg=[units(0,'mm','m'),units(0,'mm','m'),units(0,'mm','m')];
hopper.thrust_pos=[units(0,'mm','m'),units(0,'mm','m'),units(72,'mm','m')];

R_tank=units(0,'mm','m');
L_fuel=units(0,'mm','m');
hopper.geometry=[R_tank,L_fuel];

