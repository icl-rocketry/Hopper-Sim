import trusspy as tp
import numpy as np
from sympy import *
from math import *
import matplotlib.pyplot as plt

# Tip over analysis

#'''
g = 9.81
m = 150
v = 2.198
hcg = 1300 * 0.001
Ig = 2.7 * 10**10 # gmm^2 - taken from the concept 1 assembly CAD, should be accurate enough
Ig = (Ig/1000) / 1000**2  # unit conversion kgm^2
square_width = 1.4

r = sqrt(hcg**2 + (square_width/2)**2)
print("r = ", r)

I0 = Ig + m*r**2
print("I0 = ", I0)

thetadot = m*v*hcg / (Ig + m*r**2)
print("thetadot = ", thetadot)

total_energy = 0.5*I0*thetadot**2 + m*g*hcg
print("total energy = ", total_energy)

tipover_critical_energy = m*g*r
print("tipover critical energy = ", tipover_critical_energy)

v_critical = sqrt(2*m*g*(r-hcg)/I0) * (Ig + m*r**2) / (m*hcg)
print("v_critical = ", v_critical, " m/s")

tipping_angle = atan(0.5*square_width/hcg) * 180 / pi
print("tipping angle = ", tipping_angle)

max_width = sqrt(2) * square_width
print("max_width = ", max_width)

def tip_v(sq_width,h_cg):
    global Ig
    global g
    global m

    r = sqrt(h_cg**2 + (sq_width/2)**2)
    I0 = Ig + m*r**2
    tip_v = sqrt(2*g*I0*(r-h_cg)/(m*h_cg**2))
    return tip_v

def tip_angle(sq_width,h_cg):
    return atan(0.5*sq_width/hcg) * 180 / pi


#'''

#'''
sq_widths = np.arange(0.5,2,0.01)
tip_vs = [tip_v(sq_widths[i],hcg) for i in range(len(sq_widths))]
tip_angles = [tip_angle(sq_widths[i],hcg) for i in range(len(sq_widths))]

plt.xlabel("square width (m)")
plt.ylabel("critical tip velocity")
plt.title(f"Tip velocity vs width with h_cg = {hcg} m")
plt.grid(which='major', color='k', linestyle='-', linewidth=0.5)
plt.grid(which='minor', color='k', linestyle='-', linewidth=0.05)
plt.minorticks_on()
plt.plot(sq_widths,tip_vs)
#plt.show()

plt.xlabel("square width (m)")
plt.ylabel("tipping angle (degrees)")
plt.title(f"tip angle vs width with h_cg = {hcg} m")
plt.plot(sq_widths,tip_angles)
plt.grid(which='major', color='k', linestyle='-', linewidth=0.5)
plt.grid(which='minor', color='k', linestyle='-', linewidth=0.05)
plt.minorticks_on()
#plt.show()
#'''


# CAD sizing calculations


# Sizing procedure:
# 1. Determine and input desired h, d, l_top and l_main (exact) and w (approximate)
# 2. Run code and change CAD according to variables printed out 



# Approximate Leg geometry - refer to notebook or slides

xu = 120 * 0.001
yu = 50 * 0.001
zu = -300 * 0.001

yg = -600 * 0.001
zg = -1000 * 0.001

wu = 440 * 0.001
main_strut_length = sqrt(yg**2+zg**2)
side_strut_length = sqrt(xu**2+(yu-yg)**2+(zu-zg)**2)

strut_angle = atan(yg/zg) * 180 / pi

print(" ")
print("Approximate sizings below:")
print("main_strut_length = ", main_strut_length)
print("side_strut_length = ", side_strut_length)
print("strut_angle = ", strut_angle)


config = 1  #0 for long(default), 1 for short

h = 312
d = 50
w = 90
l_top = 500
l_main = [1230,1160][config]   # bolt to bolt before bolt offset
bolt_offset = [4.191,5.778][config]   # this depends on strut angle only
c_channel_hole_offset = [18.26, 21.315][config]   # this depends on strut angle only
strut_angle = [35,30][config]   

print(" ")
config = ["long", "short"][config]
print(f"Config = {config}")
print(f"CAD sizing stuff below, h = {h}, d = {d}, w = {w}, bolt offset = {bolt_offset}, top plate width = {l_top}")


x = l_main*cos((90-strut_angle)*pi/180) + d + 2*bolt_offset*cos(strut_angle*pi/180)
y = w
z = l_main*sin((90-strut_angle)*pi/180) - h - (30-c_channel_hole_offset) - 2*bolt_offset*sin(strut_angle*pi/180) 

print(f"x = {x}, y = {y}, z = {z}")
print(" ")

def ls():
    global x
    global y
    global z
    ls = sqrt(x**2 + y**2 + z**2)
    print("side_strut_cad_length = ", ls)
    return ls
    
def w_lsfix(ls):
    global x
    global y
    global z
    w_lsfix = sqrt(ls**2 - x**2 - z**2)
    print(f"w if ls equals {ls} = ", w_lsfix)
    return w_lsfix

lsfree = ls()

if (round(lsfree,0)-28-28)%1.5 == 0:  # this equality checks if the threads align i.e. after substracting two 28 mm for the rod ends, is the length a multiple of pitch 1.5
    w_ls_set = w_lsfix(round(lsfree,0))
    threaded_rod_length = round(lsfree,0) - 28 - 28
elif (round(lsfree,0)-28-28)%1.5 == 1:
    w_ls_set = w_lsfix(round(lsfree,0) - 1)
    threaded_rod_length = round(lsfree,0) - 1 - 28 - 28
else:
    w_ls_set = w_lsfix(round(lsfree,0) + 1)
    threaded_rod_length = round(lsfree,0) + 1 - 28 - 28

print(f"Rod end bracket offset = {0.5*l_top - 27 - d + 7}")
print(f"Rod end bracket width offset = {w_ls_set + 32.4}")
print("threaded rod length = ", threaded_rod_length)
rod_end_c_channel_angle = atan(w_ls_set/x) * 180 / pi
print("rod end c channel angle = ", rod_end_c_channel_angle)


# Stress and force analysis

d = 0.01
w_outer = 0.0508
w_inner = 0.0381
E = 190*10**9
UTS_6061 = 279*10**6

# put FEA results here
F_static = 1000
max_stress_rod = 24*10**6  # max value in FEA for conservative, in reality should be uniform in rod but isn't in the model
max_stress_strut = -20*10**6
max_disp = 0.968*10**(-3)
#

F_rod = max_stress_rod * pi * (d/2)**2  # conservative value
F_rod_end_fails = 10000   # SKF M10 rod end data, rod end bearing fails before thread pullout at around 13 kN
rod_SF = F_rod_end_fails/F_rod
strut_SF = UTS_6061/abs(max_stress_strut)

k = F_static/max_disp
F_shock = k*sqrt(0.5*m*1*0.2/k)   #1 is effective accel at TWR 0.9, 0.2 is height of landing procedure start
print("FEA results:")
print(f"In {F_static} N static loading, Rod = {F_rod} N, Rod SF = {rod_SF}, Strut SF = {strut_SF}")
print(f"Max shock load per leg = {F_shock} N")




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

youngs_modulus = 69 * 10**9
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
    force_scale=0.0001,
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
print(" ")
print("done")
