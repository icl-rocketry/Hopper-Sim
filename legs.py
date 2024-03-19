import trusspy as tp
import numpy as np
from sympy import *
from math import *

# Leg geometry - refer to notebook or slides

xu = 200 * 0.001
yu = 200 * 0.001
zu = -212 * 0.001

yg = -700 * 0.001
zg = -713 * 0.001

wu = 440 * 0.001

# Tip over analysis

g = 9.81
m = 150
v = 1
hcg = 1200 * 0.001
Ig = 2.7 * 10**10 # gmm^2
Ig = (Ig/1000) / 1000**2 

r = sqrt(hcg**2 + ((sqrt(2)/4) * (2*yg + wu))**2)
print("r = ", r)

I0 = Ig + m*r**2
print("I0 = ", I0)

thetadot = m*v*hcg / (Ig + m*r**2)
print("thetadot = ", thetadot)

factor = (I0*thetadot**2 + 2*m*g*hcg) / (2*m*g)
#print("factor = ", factor)

tip_factor = factor/r
print("tip_factor = ", tip_factor)

tipping_angle = atan((sqrt(2)/4) * (2*abs(yg) + wu)/hcg) * 180 / pi
print("tipping angle = ", tipping_angle)

max_width = wu + 2*abs(yg)
print("max_width = ", max_width)

square_width = max_width * sqrt(2)/2
print("square_width = ", square_width)

# sizes

main_strut_length = sqrt(yg**2+zg**2)
side_strut_length = sqrt(xu**2+(yu-yg)**2+(zu-zg)**2)

strut_angle = atan(yg/zg) * 180 / pi


print("main_strut_length = ", main_strut_length)
print("side_strut_length = ", side_strut_length)
print("strut_angle = ", strut_angle)



# Stiffness analysis using trusspy

M = tp.Model(logfile=True)

#https://trusspy.readthedocs.io/en/latest/examples/eNTA-A/eNTA-A.html

with M.Nodes as MN:  # node creation
    MN.add_node(1, coord=(0, 0, 0))
    MN.add_node(2, coord=(xu, yu, zu))
    MN.add_node(3, coord=(-xu, yu, zu))
    MN.add_node(4, coord=(0, yg, zg))

element_type = 1  # truss
material_type = 1  # linear-elastic

youngs_modulus = 69 * 10**6
A_large = pi * 0.035 ** 2 - pi * 0.030 ** 2
A_small = pi * 0.015 ** 2 - pi * 0.012 ** 2

with M.Elements as ME: # link creation (not sure gpropr is yet)
    ME.add_element(1, conn=(1, 4), geometric_properties=[A_large])#, gprop=[0.75])
    ME.add_element(2, conn=(2, 4), geometric_properties=[A_small])#, gprop=[1])
    ME.add_element(3, conn=(3, 4), geometric_properties=[A_small])#, gprop=[0.5])

    ME.assign_etype("all", element_type)
    ME.assign_mtype("all", material_type)
    ME.assign_material("all", [youngs_modulus])

with M.Boundaries as MB: # fix nodes 1,2,3
    MB.add_bound_U(1, (0, 0, 0))
    MB.add_bound_U(2, (xu, yu, zu))
    MB.add_bound_U(3, (-xu, yu, zu))

up_force = 1000 # this is not reflective of actual loading modes, it is just to find the vertical stiffness

with M.ExtForces as MF: # vertical force 
    MF.add_force(4, (0, 0, up_force))


M.Settings.dlpf = 0.005   # some settings to tweak around
M.Settings.du = 0.05
M.Settings.incs = 200
M.Settings.stepcontrol = True
M.Settings.maxfac = 4
M.Settings.ftol = 8
M.Settings.xtol = 8
M.Settings.nfev = 8
M.Settings.dxtol = 1.25

M.build()
M.run()


pinc = M.Settings.incs

fig, ax = M.plot_model(
    view="3d",
    contour="force",
    lim_scale=(-0.5, 0.5, -0.5, 0.5, -1, 0),
    force_scale=1,
    inc=pinc,
)
fig.savefig("loaded.png")

'''
M.plot_movie(
    view="3d",
    contour="force",
    lim_scale=(-0.5, 0.5, -0.5, 0.5, -1, 0),  # 3D
    # lim_scale=-5, #XZ
    # lim_scale=(-4,4,-2,6), #XY
    # lim_scale=(-2,6,-2,6), #YZ
    #cbar_limits=[-0.3, 0.3],
    force_scale=0.1/up_force,
    incs=range(0, M.Settings.incs, 20),
)
'''

# displacement of feet attachment

Disp = "Displacement Z"
fig, ax = M.plot_history(nodes=[4, 4], X=Disp, Y="Force Z")
fig.savefig("feet_Disp.png")
print("done")
