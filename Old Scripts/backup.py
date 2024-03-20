# Packages Required: numpy, scipy, sympy 
import numpy as np
import matplotlib.pyplot as plt
from sympy import *
import scipy as sc

# Symbol definition
x, y, z, xx, yy, zz, d, e, f, p, q, r, phi, theta, psi, m, Ixx, Iyy, Izz, xCG, yCG, zCG = symbols('x y z xx yy zz d e f p q r phi theta psi m Ixx Iyy Izz xCG yCG zCG')

# Trajectory Formulation
ts = 10 # Simulation time (s)
dt = 0.001 # Time step (s)
maxThrust = 3000 # Maximum Hopper Engine Thrust (N)


m0 = 150 # Initial Mass (kg)
Fp = [1000,1000, maxThrust] # Thrust Vector (N)
Fge = [0, 0, -9.81] # Gravity Vector in Earth-Fixed Axes (m/s^2)
I_initial = [27.920, 27.960, 7.87870] # Inertia Tensor (kg*m^2)
Isp = 190 # Specific Impulse (s)
rocketCOM = [4, 0, 0] # Rocket Center of Mass (m)
engineCOM = [6, 0, 0] # Engine Center of Mass (m)


momentArm = np.subtract(engineCOM, rocketCOM) # Moment Arm Vector (m)
moment = np.cross(-momentArm, Fp) # Moment Vector (N*m)
print(moment)

# Initialise state vector
x0 = np.zeros(22) # State Vector

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
x0[19] = rocketCOM[0]
x0[20] = rocketCOM[1]
x0[21] = rocketCOM[2]


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

# State Space Formulation
stateFunction = Matrix([[xx],
            [yy],
            [zz],
            [Fp[0]/m + Fgb[0] - (q*zz - r*yy)],
            [Fp[1]/m + Fgb[1] - (r*xx - p*zz)],
            [Fp[2]/m + Fgb[2] - (p*yy - q*xx)],
            [p],
            [q],
            [r],
            [(moment[0] - (q*r*(Iyy - Izz)))/Ixx],
            [(moment[1] - (p*r*(Izz - Ixx)))/Iyy],
            [(moment[2] - (p*q*(Ixx - Iyy)))/Izz],
            [p + (q*sin(phi) + r*cos(phi))*tan(theta)],
            [q*cos(phi) - r*sin(phi)],
            [(q*sin(phi) + r*cos(phi))/cos(theta)],
            [(-maxThrust/Isp) * m], # Mass Decay, need to multiply by m here otherwise Jacobian will be incorrect
            [((yCG - engineCOM[1])**2 + (zCG - engineCOM[2])**2) * -Fp[0]/Isp],
            [((xCG - engineCOM[0])**2 + (zCG - engineCOM[2])**2) * -Fp[1]/Isp],
            [((xCG - engineCOM[0])**2 + (yCG - engineCOM[1])**2) * -Fp[2]/Isp],
            [-0.1 * xCG],
            [0],
            [0]])

# Linearisation
JacobianState = stateFunction.jacobian([x, y, z, xx, yy, zz, d, e, f, p, q, r, phi, theta, psi, m, Ixx, Iyy, Izz, xCG, yCG, zCG]) # Note this should not be iterated in a loop, it should be done once only
A = lambdify(([x, y, z, xx, yy, zz, d, e, f, p, q, r, phi, theta, psi, m, Ixx, Iyy, Izz, xCG, yCG, zCG]), JacobianState, "scipy") # Convert to a function -> scipy array

nln = lambdify(([x, y, z, xx, yy, zz, d, e, f, p, q, r, phi, theta, psi, m, Ixx, Iyy, Izz, xCG, yCG, zCG]), stateFunction, "scipy")

def system(t, y):
    return A(*y) @ y

def nlnsystem(t, y):
    return np.array(nln(*y)).flatten()

def hit_ground(t, y):
    return y[2]
hit_ground.terminal = True
hit_ground.direction = -1
 
if __name__ == "__main__":
    result = sc.integrate.solve_ivp(system, [0, ts], x0, method='RK45', events=hit_ground, t_eval=np.linspace(0, ts, int(ts/dt)))
    trajectory = np.zeros((3, len(result.t)))
    # Body to Earth-Fixed Frame Transformation Matrix
    teb = lambdify(([phi, theta, psi]), Teb, "scipy")
    for i in range(len(result.t)):
        trajectory[0:3, i] = np.dot(teb(result.y[12, i], result.y[13, i], result.y[14, i]), result.y[0:3, i])




#------------------- PLOTS -------------------#

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

    # Plot 3D Trajectory in Earth-Fixed Frame
    plt.figure(3)
    ax = plt.axes(projection='3d')
    ax.plot3D(trajectory[0], trajectory[1], trajectory[2], 'gray')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.set_title('3D Trajectory in Earth-Fixed Frame')

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

    