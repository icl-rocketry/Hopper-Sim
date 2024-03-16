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
Fge = [0, 0, -9.81] # Gravity Vector in Earth-Fixed Axes (m/s^2)
I_initial = [27.920, 27.960, 7.87870] # Inertia Tensor (kg*m^2)
Isp = 190 # Specific Impulse (s)
rocketCOM = [4, 0, 0] # Rocket Center of Mass (m)
engineCOM = [6, 0, 0] # Engine Center of Mass (m)

momentArm = np.subtract(rocketCOM, engineCOM) # Moment Arm Vector (m)
moment = np.cross(momentArm, [Fpx, Fpy, Fpz]) # Moment Vector (N*m), symbolic this shit pls


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
x0 = np.zeros(22)

# Initial State Vector
x0[0] = 0
x0[1] = 0
x0[2] = 0
x0[3] = 0
x0[4] = 0
x0[5] = 0
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
x0[19] = rocketCOM[0]
x0[20] = rocketCOM[1]
x0[21] = rocketCOM[2]

"""
Control Vector:
0: Fpx - Force in body x-direction (N)
1: Fpy - Force in body y-direction (N)
2: Fpz - Force in body z-direction (N)
"""
u0 = np.zeros(3) # Control Vector

# Initial Control (Force) Vector
u0[0] = 0
u0[1] = 0
u0[2] = 3000

# https://society-of-flight-test-engineers.github.io/handbook-2013/axis-systems-and-transformations-raw.html

# Earth to Body Transformation Matrix
Tbe = Matrix([[cos(theta)*cos(phi), cos(theta)*sin(phi), -sin(theta)],
              [sin(phi)*sin(theta)*cos(psi) - cos(phi)*sin(psi), sin(phi)*sin(theta)*sin(psi) + cos(phi)*cos(psi), sin(phi)*cos(theta)],
              [sin(phi)*sin(psi) + cos(phi)*sin(theta)*cos(psi), cos(phi)*sin(theta)*sin(psi) - sin(phi)*cos(psi), cos(phi)*cos(theta)]])

# Body to Earth Transformation Matrix
Teb = Matrix([[cos(theta)*cos(phi), sin(phi)*sin(theta)*cos(psi) - cos(phi)*sin(psi), cos(phi*sin(theta)*cos(psi) + sin(phi)*sin(psi))],
             [cos(theta)*sin(phi), sin(phi)*sin(theta)*sin(psi) + cos(phi)*sin(psi), sin(phi)*sin(theta)*cos(psi) - cos(phi)*sin(psi)],
             [-sin(theta), sin(phi)*cos(theta), cos(phi)*cos(theta)]])

Fgb = np.dot(Tbe, Fge) # Gravity Vector in Body-Fixed Axes (N)
print(Fgb)   

#------------------- STATE SPACE MODEL -------------------#

# For the dynamics of the system, we have a set of non-linear differential equations that describe the motion of the hopper. *dot* equations

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
            [(-sqrt(Fpx**2+Fpy**2+Fpz**2)/Isp * m)], # Mass Decay, need to multiply by m here otherwise Jacobian will be incorrect
            [((yCG - engineCOM[1])**2 + (zCG - engineCOM[2])**2) * -sqrt(Fpx**2+Fpy**2+Fpz**2)/Isp],
            [((xCG - engineCOM[0])**2 + (zCG - engineCOM[2])**2) * -sqrt(Fpx**2+Fpy**2+Fpz**2)/Isp],
            [((xCG - engineCOM[0])**2 + (yCG - engineCOM[1])**2) * -sqrt(Fpx**2+Fpy**2+Fpz**2)/Isp],
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

# Create ananonymous function for the non-linear system
nln = lambdify(([(x, y, z, xx, yy, zz, d, e, f, p, q, r, phi, theta, psi, m, Ixx, Iyy, Izz, xCG, yCG, zCG), (Fpx, Fpy, Fpz)]), stateFunction, "scipy")

# Linearisation for LQR Control -> need to do in Earth-Fixed Frame
jA = controlFunction.jacobian([x, y, z, xx, yy, zz, d, e, f, p, q, r])
jB = controlFunction.jacobian([Fpx, Fpy, Fpz])

A = lambdify(([(x, y, z, xx, yy, zz, d, e, f, p, q, r), (Fpx, Fpy, Fpz)]), jA, "scipy")
B = lambdify(([(x, y, z, xx, yy, zz, d, e, f, p, q, r), (Fpx, Fpy, Fpz)]), jB, "scipy")

#------------------- SIMULATION -------------------#

#def linsystem(t, y, u):
#    return A(y, u) @ y + B(y, u) @ u

def nlinsystem(t, y, u):
    return nln(y, u)

# Event when the hopper hits the ground
def hit_ground(t, y):
    return y[2]

hit_ground.terminal = True # Stop the simulation when the hopper hits the ground
hit_ground.direction = -1 # Check for negative direction

# Event when the hopper has no fuel
def no_fuel(t, y):
    return y[15] - m_dry

no_fuel.terminal = True
no_fuel.direction = -1


if __name__ == "__main__":

    result = x0 # 
    controls = u0 # Control Vector
    
    # Reference Setpoints in Earth-Fixed Frame
    x_ref = 0
    y_ref = 0
    z_ref = 0 # Target Altitude
    phi_ref = 0
    theta_ref = 0
    psi_ref = 0
    ref = np.array([x_ref, y_ref, z_ref, phi_ref, theta_ref, psi_ref])
           
    for i in range(len(t)-1):
        temp = sc.integrate.solve_ivp(nlinsystem, (t[i], t[i+1]), x0, args=(u0,), method='RK45', events=(hit_ground,no_fuel))
        x0 = temp.y[:,-1]
        result = np.append(result, temp.y, axis=0)
        
        # Convert body fixed states to earth fixed states
        
    """    
        Q = np.eye(12)
        R = np.eye(3)
        K, S, E = ct.lqr(A(changeState, u0), B(changeState, u0), Q, R)  # LQR Controller

        # Control Output Calculation
        error = np.array([result[i] - ref[i] for i in range(len(result))])
        u = -K @ error # Add anti-windup here?
        
        # Control Output Saturation
        u = np.clip(u, 0.3*maxThrust, maxThrust)

        # Update Control Input
        u0[0] = u[0]
        u0[1] = u[1]
        u0[2] = u[2]
    """    
    # END FOR LOOP

    
    
# Plot step responses

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