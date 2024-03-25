# Packages Required: numpy, scipy, sympy, matplotlib, control
import numpy as np
import matplotlib.pyplot as plt
from sympy import *
import scipy as sc
import control as ct

# Symbol definition
x, y, z, xx, yy, zz, d, e, f, p, q, r, phi, theta, psi, m, Ixx, Iyy, Izz, xCG, yCG, zCG = symbols('x y z xx yy zz d e f p q r phi theta psi m Ixx Iyy Izz xCG yCG zCG')
Fpx, Fpy, Fpz = symbols('Fpx Fpy Fpz')

# Trajectory Formulation
ts = 10 # Simulation time (s)
dt = 0.01 # Time step (s)
t = np.linspace(0, ts, int(ts/dt)) # Time Vector

maxThrust = 5000 # Maximum Hopper Engine Thrust (N)
m0 = 200 # Initial Mass (kg)
m_dry = 60 # Dry Mass (kg)
Fp = [maxThrust/sqrt(3), maxThrust/sqrt(3), maxThrust/sqrt(3)] # Thrust Vector (N)
Fge = [0, 0, -9.81] # Gravity Vector in Earth-Fixed Axes (m/s^2)
I_initial = [27.920, 27.960, 7.87870] # Inertia Tensor (kg*m^2)
Isp = 190 # Specific Impulse (s)


x0 = np.zeros(22) # State Vector
u0 = np.zeros(3) # Control Vector


""" State Vector:
0: x - Longitudinal Position in Body-Fixed Axes (m)
1: y - Lateral Position in Body-Fixed Axes (m)
2: z - Vertical Position in Body-Fixed Axes (m)
3: xx - Longitudinal Velocity in Body-Fixed Axes (m/s)
4: yy - Lateral Velocity in Body-Fixed Axes (m/s)
5: zz - Vertical Velocity in Body-Fixed Axes (m/s)
6: d - Roll Angle in Body-Fixed Axes (rad) REMOVE
7: e - Pitch Angle in Body-Fixed Axes (rad) REMOVE
8: f - Yaw Angle in Body-Fixed Axes (rad) REMOVE
9: p - Roll Rate (rad/s)
10: q - Pitch Rate (rad/s)
11: r - Yaw Rate (rad/s)
12: phi - Roll Euler Angle in Earth-Fixed Axes (rad)
13: theta - Pitch Euler Angle in Earth-Fixed Axes (rad)
14: psi - Yaw Euler Angle in Earth-Fixed Axes (rad)
15: m - Mass (kg)
16: Ixx - Inertia Tensor (kg*m^2)
17: Iyy - Inertia Tensor (kg*m^2)
18: Izz - Inertia Tensor (kg*m^2)
19: xCG - x-Coordinate of the Center of Gravity (m)
20: yCG - y-Coordinate of the Center of Gravity (m)
21: zCG - z-Coordinate of the Center of Gravity (m)
"""

# Initial Conditions
x0[0] = 0
x0[1] = 0
x0[2] = 0
x0[3] = 0
x0[4] = 0
x0[5] = 10
x0[6] = 0
x0[7] = 0
x0[8] = 0
x0[9] = 0
x0[10] = 0
x0[11] = 0
x0[12] = 0
x0[13] = 0
x0[14] = 0
x0[15] = m0
x0[16] = I_initial[0]
x0[17] = I_initial[1]
x0[18] = I_initial[2]
x0[19] = 4
x0[20] = 0
x0[21] = 0


u0[0] = Fp[0]
u0[1] = Fp[1]
u0[2] = Fp[2]


# Earth to Body-Fixed Frame Transformation Matrix
Tbe = Matrix([[cos(theta)*cos(phi), sin(phi)*sin(theta)*cos(psi) - cos(phi)*sin(psi), cos(phi*sin(theta)*cos(psi) + sin(phi)*sin(psi))],
             [cos(theta)*sin(phi), sin(phi)*sin(theta)*sin(psi) + cos(phi)*sin(psi), sin(phi)*sin(theta)*cos(psi) - cos(phi)*sin(psi)],
             [-sin(theta), sin(phi)*cos(theta), cos(phi)*cos(theta)]])

Teb = lambdify([phi, theta, psi], Tbe.inv(), "scipy")

Fgb = np.dot(Tbe, Fge) # Gravity Vector in Body-Fixed Axes (N)
earthTraj = np.dot(Tbe.inv(), x0[0:3]) # Earth-Fixed Trajectory

rocketCOM = Matrix([xCG, yCG, zCG]) # Rocket Center of Mass (m)
engineCOM = [10, 0, 0] # Engine Center of Mass (m) assume constant

momentArm = np.subtract(engineCOM, rocketCOM) # Moment Arm Vector (m)
moment = np.cross(momentArm, [Fpx,Fpy,Fpz]) # Moment Vector (N*m)       
      
#------------------- STATE SPACE MODEL -------------------#

# For the dynamics of the system, we have a set of non-linear differential equations that describe the motion of the hopper.

stateFunction = Matrix([[xx],
            [yy],
            [zz],
            [Fpx/m + Fgb[0] - (q*zz - r*yy)],
            [Fpy/m + Fgb[1] - (r*xx - p*zz)],
            [Fpz/m + Fgb[2] - (p*yy - q*xx)],
            [p],
            [q],
            [r],
            [(moment[0] - (q*r*(Izz - Iyy)))/Ixx],
            [(moment[1] - (p*r*(Izz - Ixx)))/Iyy],
            [(moment[2] - (p*q*(Ixx - Iyy)))/Izz],
            [p + (q*sin(phi) + r*cos(phi))*tan(theta)],
            [q*cos(phi) - r*sin(phi)],
            [(q*sin(phi) + r*cos(phi))/cos(theta)],
            [(-maxThrust/Isp * m)], # Mass Decay, need to multiply by m here otherwise Jacobian will be incorrect
            [((yCG - engineCOM[1])**2 + (zCG - engineCOM[2])**2) * -Fpx/Isp],
            [((xCG - engineCOM[0])**2 + (zCG - engineCOM[2])**2) * -Fpy/Isp],
            [((xCG - engineCOM[0])**2 + (yCG - engineCOM[1])**2) * -Fpz/Isp],
            [-0.1 * xCG],
            [0],
            [0]])

# However, for control purposes, we need to linearise the system around a set of operating points.

controlFunction = Matrix([[xx],
            [yy],
            [zz],
            [Fpx/m + Fgb[0] - (q*zz - r*yy)],
            [Fpy/m + Fgb[1] - (r*xx - p*zz)],
            [Fpz/m + Fgb[2] - (p*yy - q*xx)],
            [p],
            [q],
            [r],
            [(moment[0] - (q*r*(Izz - Iyy)))/Ixx],
            [(moment[1] - (p*r*(Izz - Ixx)))/Iyy],
            [(moment[2] - (p*q*(Ixx - Iyy)))/Izz],
            [p + (q*sin(phi) + r*cos(phi))*tan(theta)],
            [q*cos(phi) - r*sin(phi)],
            [(q*sin(phi) + r*cos(phi))/cos(theta)]])

# Maintain non-linearity for simulation
nln = lambdify(([(x, y, z, xx, yy, zz, d, e, f, p, q, r, phi, theta, psi, m, Ixx, Iyy, Izz, xCG, yCG, zCG), (Fpx, Fpy, Fpz)]), stateFunction, "scipy")

# Linearisation for LQR Control
JacobianState = stateFunction.jacobian([x, y, z, xx, yy, zz, d, e, f, p, q, r, phi, theta, psi, m, Ixx, Iyy, Izz, xCG, yCG, zCG])
JacobianControl = stateFunction.jacobian([Fpx, Fpy, Fpz])

A = lambdify(([(x, y, z, xx, yy, zz, d, e, f, p, q, r, phi, theta, psi, m, Ixx, Iyy, Izz, xCG, yCG, zCG), (Fpx, Fpy, Fpz)]), JacobianState, "scipy") # Convert to a function -> scipy array
B = lambdify(([(x, y, z, xx, yy, zz, d, e, f, p, q, r, phi, theta, psi, m, Ixx, Iyy, Izz, xCG, yCG, zCG), (Fpx, Fpy, Fpz)]), JacobianControl, "scipy") # Convert to a function -> scipy array

#------------------- SIMULATION -------------------#

def system(t, y, u):
    return A(y, u) @ y + B(y, u) @ u

def nlnsystem(t, y, u):
    return nln(y, u)

# Event when the hopper hits the ground
def hit_ground(t, y, u):
    return y[2]

hit_ground.terminal = True # Stop the simulation when the hopper hits the ground
hit_ground.direction = -1 # Check for negative direction

# Event when the hopper has no fuel
def no_fuel(t, y, u):
    return y[15] - m_dry

no_fuel.terminal = True
no_fuel.direction = -1

 
# Controller initialisation

# Saturation



if __name__ == "__main__":
    # Control stuff -> new thrust -> replacae thrust values       
    temp = sc.integrate.solve_ivp(system, [0, ts], x0, method='RK45', args=(u0,), events=(hit_ground,no_fuel), t_eval=np.linspace(0, ts, int(ts/dt)))
    
    # END FOR LOOP

    # Convert body fixed to earth fixed
    


#------------------- PLOTS -------------------#
"""
    titles = ["x", "y", "z", "xx", "yy", "zz", "d", "e", "f", "p", "q", "r", "phi", "theta", "psi", "m", "Ixx", "Iyy", "Izz", "xCG", "yCG", "zCG"]
    
    # 3D Trajectory
    plt.figure(1)
    ax = plt.axes(projection='3d')
    ax.plot3D(result.y[0], result.y[1], result.y[2], 'gray')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.set_title('3D Trajectory')
    
    # Velocity Plots
    plt.figure(2)
    for i in range(3,6):
        plt.subplot(3, 1, i-2)
        plt.title(titles[i])
        plt.plot(result.t, result.y[i])
    
    plt.figure(3)
    ax = plt.axes(projection='3d')
    ax.plot3D(result.y[6], result.y[7], result.y[8], 'gray')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.set_title('3D Attitude (Euler Angles)')   
    # Earth Fixed Euler Angles
    plt.figure(4)
    for i in range(6, 9):
        plt.subplot(3, 1, i-5)
        plt.title(titles[i])
        plt.plot(result.t, result.y[i])
    
    # Angular Rates
    plt.figure(5)
    for i in range(9, 12):
        plt.subplot(3, 1, i-8)
        plt.title(titles[i])
        plt.plot(result.t, result.y[i])
    
    # Body Fixed Euler Angles
    plt.figure(6)
    for i in range(12, 15):
        plt.subplot(3, 1, i-11)
        plt.title(titles[i])
        plt.plot(result.t, result.y[i])
    
    # Inertia Tensor
    plt.figure(7)
    for i in range(16, 19):
        plt.subplot(3, 1, i-15)
        plt.title(titles[i])
        plt.plot(result.t, result.y[i])
    
    # Center of Gravity 
    plt.figure(8)
    for i in range(19, 22):
        plt.subplot(3, 1, i-18)
        plt.title(titles[i])
        plt.plot(result.t, result.y[i])
    
    # Mass
    plt.figure(9)
    plt.plot(result.t, result.y[15])
    plt.title("Mass Decay")
    
    plt.show()
"""


#------------------- THE ABYSS -------------------#
"""
nlnstateFunction = Matrix([[xx],
            [yy],
            [zz],
            [(Fp[0]+Fgb[0])/m - (q*zz - r*yy)],
            [(Fp[1]+Fgb[1])/m - (r*xx - p*zz)],
            [(Fp[2]+Fgb[2])/m - (p*yy - q*xx)],
            [p],
            [q],
            [r],
            [(moment[0] - (q*r*(Iyy - Izz)))/Ixx],
            [(moment[1] - (p*r*(Izz - Ixx)))/Iyy],
            [(moment[2] - (p*q*(Ixx - Iyy)))/Izz],
            [p + (q*sin(phi) + r*cos(phi))*tan(theta)],
            [q*cos(phi) - r*sin(phi)],
            [(q*sin(phi) + r*cos(phi))/cos(theta)],
            [(-maxThrust/Isp)], # Mass Decay, need to multiply by m here otherwise Jacobian will be incorrect
            [((yCG - engineCOM[1])**2 + (zCG - engineCOM[2])**2) * Fp[0]],
            [((xCG - engineCOM[1])**2 + (zCG - engineCOM[2])**2) * Fp[1]],
            [((xCG - engineCOM[1])**2 + (yCG - engineCOM[2])**2) * Fp[2]],
            [-0.1],
            [0],
            [0]])

nonLinear = lambdify(([x, y, z, xx, yy, zz, d, e, f, p, q, r, phi, theta, psi, m, Ixx, Iyy, Izz, xCG, yCG, zCG]), nlnstateFunction, "scipy")

def nlnsystem(t,y):
    return nonLinear(*y)

"""