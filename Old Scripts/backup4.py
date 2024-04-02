# Packages Required: numpy, matplotlib, sympy, scipy, control
import sys
import numpy as np
import matplotlib.pyplot as plt
from casadi import *
from sympy import *
import scipy as sc
import control as ct

# Symbol definition
x, y, z, xx, yy, zz, d, e, f, p, q, r, phi, theta, psi, m, Ixx, Iyy, Izz, xCG, yCG, zCG = symbols('x y z xx yy zz d e f p q r phi theta psi m Ixx Iyy Izz xCG yCG zCG')
Fpx, Fpy, Fpz = symbols('Fpx Fpy Fpz')

# Trajectory Formulation
ts = 40 # Simulation time (s)
dt = 0.001 # Time step (s)
t = np.linspace(0, ts, int(ts/dt)) # Time Vector

maxThrust = 2500 # Maximum Hopper Engine Thrust (N)
m0 = 125 # Initial Mass (kg)
m_dry = 100 # Dry Mass (kg)
Fge = [0, 0, -9.81] # Gravity Vector in Earth-Fixed Axes (m/s^2)
I_initial = [27.920, 27.960, 7.87870] # Inertia Tensor (kg*m^2)
Isp = 190 # Specific Impulse (s)
rocketCOM = [1.2, 0, 0] # Rocket Center of Mass (m)
engineCOM = [2, 0, 0] # Engine Center of Mass (m)

momentArm = np.subtract(rocketCOM, engineCOM) # Moment Arm Vector (m)
moment = np.cross(momentArm, [Fpx, Fpy, Fpz]) # Moment Vector (N*m), symbolic this shit pls


#------------------- STATE/CONTROL VARIABLES -------------------#

""" State Vector:
0: x - Longitudinal Position in Body-Fixed Axes (m)
1: y - Lateral Position in Body-Fixed Axes (m)
2: z - Vertical Position in Body-Fixed Axes (m)
3: xx - Longitudinal Velocity in Body-Fixed Axes (m/s)
4: yy - Lateral Velocity in Body-Fixed Axes (m/s)
5: zz - Vertical Velocity in Body-Fixed Axes (m/s)
6: d - Roll Angle in Body-Fixed Axes (rad) 
7: e - Pitch Angle in Body-Fixed Axes (rad) 
8: f - Yaw Angle in Body-Fixed Axes (rad)
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
u0 = [0, 0, 2500]


#------------------- ROTATIONAL MATRIX -------------------#

# https://society-of-flight-test-engineers.github.io/handbook-2013/axis-systems-and-transformations-raw.html
# https://www.aircraftflightmechanics.com/EoMs/EulerTransforms.html

# Earth to Body Transformation Matrix
Teb = Matrix([[cos(theta)*cos(psi), sin(phi)*sin(theta)*cos(psi) - cos(phi)*sin(psi), cos(phi)*sin(theta)*cos(psi) + sin(phi)*sin(psi)],
             [cos(theta)*sin(psi), sin(phi)*sin(theta)*sin(psi) + cos(phi)*cos(psi), cos(phi)*sin(theta)*cos(psi) - sin(phi)*cos(psi)],
             [-sin(theta), sin(phi)*cos(theta), cos(phi)*cos(theta)]])
             
#Tbe = Matrix([[cos(theta)*cos(phi), cos(theta)*sin(phi), -sin(theta)],
#             [-cos(phi)*sin(psi) + sin(phi)*sin(theta)*cos(psi), cos(phi)*cos(psi) + sin(phi)*sin(theta)*sin(psi), sin(phi)*cos(theta)],
#             [sin(phi)*sin(psi) + cos(phi)*sin(theta)*cos(psi), -sin(phi)*cos(psi) + cos(phi)*sin(theta)*sin(psi), cos(phi)*cos(theta)]])

# By orthogonality, Tbe = Teb^-1
Tbe = Teb.T
Fgb = np.dot(Teb,Fge) # Gravity Vector in Body-Fixed Axes (N)


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
            [(moment[1] - (r*p*(Ixx - Izz)))/Iyy],
            [(moment[2] - (p*q*(Iyy - Ixx)))/Izz],
            [p + (q*sin(phi) + r*cos(phi))*tan(theta)],
            [q*cos(phi) - r*sin(phi)],
            [(q*sin(phi) + r*cos(phi))/cos(theta)],
            [-sqrt(Fpx**2+Fpy**2+Fpz**2)/Isp], # Mass Decay, need to multiply by m here otherwise Jacobian will be incorrect
            [0],
            [0],
            [0],
            [0],
            [0],
            [0]])

"""
Rate of change of CG and Inertias 
[((yCG - engineCOM[1])**2 + (zCG - engineCOM[2])**2) * -sqrt(Fpx**2+Fpy**2+Fpz**2)/Isp],
            [((xCG - engineCOM[0])**2 + (zCG - engineCOM[2])**2) * -sqrt(Fpx**2+Fpy**2+Fpz**2)/Isp],
            [((xCG - engineCOM[0])**2 + (yCG - engineCOM[1])**2) * -sqrt(Fpx**2+Fpy**2+Fpz**2)/Isp],
            [-0.1 * xCG],
            [0],
            [0]
"""
# However, for control purposes, we need to linearise the system around a set of operating points.

controlFunction = Matrix([[zz],
            [(moment[0] - (q*r*(Izz - Iyy)))/Ixx], 
            [(moment[1] - (r*p*(Ixx - Izz)))/Iyy],
            [(moment[2] - (p*q*(Iyy - Ixx)))/Izz],
            [p + (q*sin(phi) + r*cos(phi))*tan(theta)],
            [q*cos(phi) - r*sin(phi)],
            [(q*sin(phi) + r*cos(phi))/cos(theta)]])

# Create ananonymous function for the non-linear system to be solved by the integrator
nln = lambdify(([(x, y, z, xx, yy, zz, d, e, f, p, q, r, phi, theta, psi, m, Ixx, Iyy, Izz, xCG, yCG, zCG), (Fpx, Fpy, Fpz)]), stateFunction, "scipy")

# Linearisation for LQR Control
jA = controlFunction.jacobian([z, p, q, r, phi, theta, psi])
jB = controlFunction.jacobian([Fpx, Fpy, Fpz])

A = lambdify(([(x, y, z, xx, yy, zz, d, e, f, p, q, r, phi, theta, psi, m, Ixx, Iyy, Izz, xCG, yCG, zCG), (Fpx, Fpy, Fpz)]), jA, "scipy")
B = lambdify(([(x, y, z, xx, yy, zz, d, e, f, p, q, r, phi, theta, psi, m, Ixx, Iyy, Izz, xCG, yCG, zCG), (Fpx, Fpy, Fpz)]), jB, "scipy")

rowA, colA = jA.shape # Obtain dimensions of Jacobian Dynamics Matrix
rowB, colB = jB.shape # Obtain dimensions of Jacobian Control Matrix

C = np.diag([0, 0, 1,]) # Control Matrix


#------------------- SIMULATION -------------------#

#def linsystem(t, y, u):
#   return np.array(A(y, u) @ y + B(y, u) @ u).flatten()

def nlinsystem(t, y, u):
    """
    Non-linear system of equations for the hopper dynamics. An anonymous function is created to solve the system of equations above
    """
    return np.array(nln(y,u)).flatten()

outOfFuel = False

def thrustDecay(u, decayTime, dt, s, i):
    """
    This function simulates the linear decay of thrust output as the hopper runs out of fuel.
    ------------------------------------------------------------------------------------------------
    u: Thrust Vector before mass depletion
    decayTime: Time to reach zero-thrust(s)
    ------------------------------------------------------------------------------------------------
    """
    if (i -s) >= decayTime:
        outOfFuel = True
        return np.array([0, 0, 0])
    else:
        return np.array([u[0] - u[0]*i*dt/(decayTime), u[1] - u[1]*i*dt/(decayTime), u[2] - u[2]*i*dt/(decayTime)])


# Controllability Check
def controllabilityCheck(A, B):
    
    #NOTE: Theta (pitch) is not controllable
    
    C = B
    for i in range(1,len(A)):
        C = np.hstack((C, A**i @ B))

    rank = np.linalg.matrix_rank(C)
    size = len(A)
    
    if rank != size:
        print("Rank of Controllability Matrix: " + str(rank) + "\n"
              "Size of Matrix A: " + str(size) + "\n"
              "System is not controllable!")
        sys.exit()
        
    return 0

# Observerability Check
def observerabilityCheck(A, C):
    
    O = C
    for i in range(1,len(A)):
        O = np.vstack((O, C @ (A**i)))

    rank = np.linalg.matrix_rank(O) # Not working rn
    size = len(A)
    
    if rank != size:
        print("Rank of Observability Matrix: " + str(rank) + "\n"
              "Size of Matrix A: " + str(size) + "\n"
              "System is not observable!")
        sys.exit()
        
    return 0

"""

"""


if __name__ == "__main__":

    result = x0 # Result Vector
    controls = u0 # Control Vector
    
    # Reference Setpoints in Earth-Fixed Frame
    z_ref = 1 # Target Altitude: 1 m above ground
    p_ref = 0
    q_ref = 0
    r_ref = 0
    phi_ref = 0
    theta_ref = 0
    psi_ref = 0
    
    ref = np.array([z_ref, q_ref, r_ref, phi_ref, theta_ref, psi_ref])
    
    # Use Bryson's rule to find optimal Q and R       
    Q = np.diag([5, 1, 1, 1, 1, 1, 1]) #NOTE: Maximum allowable value = 0 -> Q_i = 1
    R = np.eye(3) 

    
    """
    [FUTURE] Hover Control:
    The flight should be divided into three phases:
    1. Ascent: Ascend to 1m in 2 seconds
    2. Hover: 10 second hover at 1m (1s buffer)
    3. Descent: Safely land at 0m in 2 seconds
    
    """
    
    ascentTime = 2
    hoverTime = 11
    descentTime = 2
    
    ascent = True
    hover = False
    descent = False
  
    def ascentControl(x0):
        
        return 0

    def hoverControl():
        return 0

    def descentControl():
        return 0
  
    s = 0
    
#------------------- SIMULATION LOOP -------------------#
    
    for i in range(len(t)-1):
        
        temp = sc.integrate.solve_ivp(nlinsystem, (t[i], t[i+1]), x0, args=(u0,), method='RK45', t_eval=[t[i+1]])
        x0 = temp.y[:,-1] # Update State Vector
        
        # Convert body fixed states to earth fixed states
        controlState = np.concatenate((x0[2:3], x0[10:12], x0[12:15]))

        
        # Check if hopper has hit the ground, exit simulation if so
        if (x0[2] < 0):
           print("Hopper has hit the ground.")
           break
        
        # Check if all fuel is depleted, control if not
        if x0[15] >= m_dry:
            
            if (t[i] >= 2):
                
                # Check Controllability
                controllabilityCheck(A(x0, u0), B(x0, u0))
                #observerabilityCheck(A(x0, u0), C)
                
                K, _, _ = ct.lqr(A(x0, u0), B(x0, u0), Q, R)  # LQR Controller to find optimal K for given Q and R
                
                # Control Output Calculation
                error = np.array([controlState[i] - ref[i] for i in range(len(controlState))])   
                u = -K @ error.T # Add anti-windup here       
            
                # Control Output Saturation
                u = np.clip(u, 0.3*maxThrust, maxThrust)

                # Update Control Input
                u0 = [u[0], u[1], u[2]]  

            else:
                if (s == 0):
                    s = i # Store counter for fuel depletion
                if (outOfFuel == False):
                    u0 = thrustDecay(u0, 3, dt, s, i) # Simulate thrust decay
                else:
                    u0 = [0, 0 ,0]
                    
            controls = np.vstack((controls, u0)) # Store Control Inputs
            
        result = np.vstack((result, x0)) # Store State Vector
        
    
    # END FOR LOOP


#------------------- PLOTS -------------------#

    titles = ["x", "y", "z", "xx", "yy", "zz", "d", "e", "f", "p", "q", "r", "phi", "theta", "psi", "m", "Ixx", "Iyy", "Izz", "xCG", "yCG", "zCG"] # Titles for plots
    
    if (len(result) != len(t)):
        t = t[0:len(result)] # In the event the simulation ends early
    
    # 3D Trajectory
    plt.figure(1)
    ax = plt.axes(projection='3d')
    ax.plot3D(result[:,0], result[:,1], result[:,2], 'gray')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.set_title('3D Trajectory')

    """
    # Velocity Plots
    plt.figure(2)
    for i in range(3,6):
        plt.subplot(3, 1, i-2)
        plt.title(titles[i])
        plt.plot(t, result[:,i])
        
        
    plt.figure(3)
    ax = plt.axes(projection='3d')
    ax.plot3D(result[:,6], result[:,7], result[:,8], 'gray')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.set_title('3D Attitude (Euler Angles)')   
    
    
    # Earth Fixed Euler Angles
    plt.figure(4)
    for i in range(6, 9):
        plt.subplot(3, 1, i-5)
        plt.title(titles[i])
        plt.plot(t, result[:,i])
    
    # Angular Rates
    plt.figure(5)
    for i in range(9, 12):
        plt.subplot(3, 1, i-8)
        plt.title(titles[i])
        plt.plot(t, result[:,i])
    
    # Body Fixed Euler Angles
    plt.figure(6)
    for i in range(12, 15):
        plt.subplot(3, 1, i-11)
        plt.title(titles[i])
        plt.plot(t, result[:,i])
    
    # Inertia Tensor
    plt.figure(7)
    for i in range(16, 19):
        plt.subplot(3, 1, i-15)
        plt.title(titles[i])
        plt.plot(t, result[:,i])
    
    # Center of Gravity 
    plt.figure(8)
    for i in range(19, 22):
        plt.subplot(3, 1, i-18)
        plt.title(titles[i])
        plt.plot(t, result[:,i])
    """
    
    # Mass
    plt.figure(9)
    plt.plot(t, result[:,15])
    plt.title("Mass Decay")
    
 
    titles2 = ["Fpx", "Fpy", "Fpz"]
 
    plt.figure(10)
    for i in range(0,3):
        plt.subplot(3, 1, i+1)
        plt.title(titles2[i])
        plt.plot(t, controls[:,i])
    
    
    plt.figure(11)
    plt.plot(t, z_ref - result[:,2])
    
    plt.show()