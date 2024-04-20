#------------------- PACKAGES -------------------#
# Packages Required: numpy, matplotlib, sympy, scipy, control, filterpy
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from sympy import *
import scipy as sc
import control as ct
from filterpy.kalman import KalmanFilter

# User defined functions
import anim as anim
import results as results
import controller as controller

# ------------------- GUI -------------------#
"""
import GUI as gui # Import GUI class from GUI.py

menu = gui.GUI()
menu.mainloop()
"""

#------------------- SYMBOLIC VARIABLES -------------------#
x, y, z, xx, yy, zz,  p, q, r, phi, theta, psi, m, Ixx, Iyy, Izz, xCG, yCG, zCG = symbols('x y z xx yy zz p q r phi theta psi m Ixx Iyy Izz xCG yCG zCG')
Fpx, Fpy, Fpz = symbols('Fpx Fpy Fpz')


#------------------- ROCKET PROPERTIES -------------------#

maxThrust = 2500 # Maximum Hopper Engine Thrust (N)
m0 = 125 # Initial Mass (kg)
m_dry = 100 # Dry Mass (kg)
Fge = [0, 0, -9.81] # Gravity Vector in Earth-Fixed Axes (m/s^2) NOTE; Issue was here lmao
I_initial = [27.920, 27.960, 7.87870] # Inertia Tensor (kg*m^2)
Isp = 190 # Specific Impulse (s)
g0 = 9.81 # Constant Gravitational Acceleration (m/s^2)
engineCOM = [2, 0, 0] # Engine Center of Mass (m)


#------------------- STATE/CONTROL VARIABLES -------------------#

""" State Vector:
0: x - Longitudinal Position in Body-Fixed Axes (m)
1: y - Lateral Position in Body-Fixed Axes (m)
2: z - Vertical Position in Body-Fixed Axes (m)
3: xx - Longitudinal Velocity in Body-Fixed Axes (m/s)
4: yy - Lateral Velocity in Body-Fixed Axes (m/s)
5: zz - Vertical Velocity in Body-Fixed Axes (m/s)
6: p - Roll Rate in Body-Fixed Axes (rad/s)
7: q - Pitch Rate in Body-Fixed Axes (rad/s)
8: r - Yaw Rate in Body-Fixed Axes (rad/s)
9: phi - Roll Euler Angle in Earth-Fixed Axes (rad)
10: theta - Pitch Euler Angle in Earth-Fixed Axes (rad) -> from Vertical
11: psi - Yaw Euler Angle in Earth-Fixed Axes (rad)
12: m - Mass (kg)
13: Ixx - Inertia Tensor (kg*m^2)
14: Iyy - Inertia Tensor (kg*m^2)
15: Izz - Inertia Tensor (kg*m^2)
16: xCG - x-Coordinate of the Center of Gravity (m)
17: yCG - y-Coordinate of the Center of Gravity (m)
18: zCG - z-Coordinate of the Center of Gravity (m)
"""

x0 = np.zeros(19)

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
x0[10] = np.pi/2
x0[11] = 0
x0[12] = m0
x0[13] = I_initial[0]
x0[14] = I_initial[1]
x0[15] = I_initial[2]
x0[16] = 1.2
x0[17] = 0
x0[18] = 0

""" Control Vector:
0: Fpx - Force in body x-direction (N)
1: Fpy - Force in body y-direction (N)
2: Fpz - Force in body z-direction (N)
"""

u0 = np.zeros(3) 

# Initial Control (Force) Vector in BODY-FIXED Axes
u0 = [2500, 0, 0]


#------------------- ROTATIONAL MATRIX -------------------#
# CHECK AXES DEFINITION IN DIFFERENT REFERENCE FRAMES
# Reference: https://www.intechopen.com/chapters/64567 

# Earth to Body Transformation Matrix
Teb = Matrix([[cos(theta)*cos(psi), sin(phi)*sin(theta)*cos(psi) - cos(phi)*sin(psi), cos(phi)*sin(theta)*cos(psi) + sin(phi)*sin(psi)],
            [cos(theta)*sin(psi), sin(phi)*sin(theta)*sin(psi) + cos(phi)*cos(psi), cos(phi)*sin(theta)*cos(psi) - sin(phi)*cos(psi)],
            [-sin(theta), sin(phi)*cos(theta), cos(phi)*cos(theta)]])
             
Tbe = Teb.T # By orthogonality, Tbe = Teb^-1

#Tbe = Matrix([[cos(theta)*cos(phi), cos(theta)*sin(phi), -sin(theta)],
#             [-cos(phi)*sin(psi) + sin(phi)*sin(theta)*cos(psi), cos(phi)*cos(psi) + sin(phi)*sin(theta)*sin(psi), sin(phi)*cos(theta)],
#             [sin(phi)*sin(psi) + cos(phi)*sin(theta)*cos(psi), -sin(phi)*cos(psi) + cos(phi)*sin(theta)*sin(psi), cos(phi)*cos(theta)]])

Fgb = np.dot(Teb,Fge) # Gravity Vector in Body-Fixed Axes (N)

rocketCOM = [xCG, yCG, zCG] # Rocket Center of Mass (m), symoblically defined
momentArm = np.subtract(rocketCOM, engineCOM) # Moment Arm Vector (m)
moment = np.cross(momentArm, [Fpx, Fpy, Fpz]) # Moment Vector (N*m), symoblically defined

earthTrajMath = Matrix(np.dot(Tbe, [x, y, z])) # Trajectory in Earth-Fixed Axes
earthTrajFunc = lambdify([x, y, z, phi, theta, psi], earthTrajMath, "scipy")


#------------------- STATE SPACE MODEL -------------------#

# For the dynamics of the system, we have a set of non-linear differential equations that describe the motion of the hopper. *dot* equations

stateFunction = Matrix([[xx],
            [yy],
            [zz],
            [Fpx/m + Fgb[0] - (q*zz - r*yy)], 
            [Fpy/m + Fgb[1] - (r*xx - p*zz)],
            [Fpz/m + Fgb[2] - (p*yy - q*xx)],
            [(moment[0] - (q*r*(Izz - Iyy)))/Ixx], 
            [(moment[1] - (r*p*(Ixx - Izz)))/Iyy],
            [(moment[2] - (p*q*(Iyy - Ixx)))/Izz],
            [p + (q*sin(phi) + r*cos(phi))*tan(theta)],
            [q*cos(phi) - r*sin(phi)],
            [(q*sin(phi) + r*cos(phi))/cos(theta)],
            [-sqrt(Fpx**2 + Fpy**2 + Fpz**2)/(Isp*g0)],
            [((yCG - engineCOM[1])**2 + (zCG - engineCOM[2])**2) * -sqrt(Fpx**2+Fpy**2+Fpz**2)/(Isp*g0)],
            [((xCG - engineCOM[0])**2 + (zCG - engineCOM[2])**2) * -sqrt(Fpx**2+Fpy**2+Fpz**2)/(Isp*g0)],
            [((xCG - engineCOM[0])**2 + (yCG - engineCOM[1])**2) * -sqrt(Fpx**2+Fpy**2+Fpz**2)/(Isp*g0)],
            [-0.02],
            [0],
            [0]])


# Create ananonymous function for the non-linear system to be solved by the integrator
nln = lambdify(([(x, y, z, xx, yy, zz, p, q, r, phi, theta, psi, m, Ixx, Iyy, Izz, xCG, yCG, zCG), (Fpx, Fpy, Fpz)]), stateFunction, "scipy")

# Linearisation for LQR Control
jA = stateFunction.jacobian([x, y, z, xx, yy, zz, p, q, r, phi, theta, psi, m, Ixx, Iyy, Izz, xCG, yCG, zCG])
jB = stateFunction.jacobian([Fpx, Fpy, Fpz])

A = lambdify(([(x, y, z, xx, yy, zz, p, q, r, phi, theta, psi, m, Ixx, Iyy, Izz, xCG, yCG, zCG), (Fpx, Fpy, Fpz)]), jA, "scipy")
B = lambdify(([(x, y, z, xx, yy, zz, p, q, r, phi, theta, psi, m, Ixx, Iyy, Izz, xCG, yCG, zCG), (Fpx, Fpy, Fpz)]), jB, "scipy")

rowA, colA = jA.shape # Obtain dimensions of Jacobian Dynamics Matrix
rowB, colB = jB.shape # Obtain dimensions of Jacobian Control Matrix

xEq = [x, y, 1, 0, 0, 0, 0, 0, 0, 0, np.pi/2, 0, m0, I_initial[0], I_initial[1], I_initial[2], 1.2, 0, 0] # Equilibrium States
uEq = [m0*9.81, 0, 0] # Equilibrium Controls

C = np.diag([0, 0, 1, 
             0, 0, 0, 
             0, 0, 0, 
             0, 0, 0,
             0, 0, 0,
             0, 0, 0, 0]) # Control Matrix


#------------------- SIMULATION -------------------#

#def linsystem(t, y, u):
#   return np.array(A(y, u) @ y + B(y, u) @ u).flatten()

def nlinsystem(t, y, u):
    return np.array(nln(y,u)).flatten()


#---------------- CONTROLLER SETUP -----------------#

# Reference Setpoints in Earth-Fixed Frame
z_ref = 1 # Target Altitude: 1 m above ground, this needs to be in ECEF (NOTE: in body-fixed, bodyX is initially aligned with Earth-Z)
q_ref = 0
r_ref = 0
theta_ref = np.pi/2
psi_ref = 0

ref = np.array([z_ref, q_ref, r_ref, theta_ref, psi_ref])

# Use Bryson's rule to find optimal Q and R       
Q = np.diag([0, 0, 10, 0, 0, 0, 0, 1, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0]) #NOTE: Maximum allowable value = 0 -> Q_i = 1
R = np.eye(3) 

ascentTime = 2
hoverTime = 11
descentTime = 2


def flightControl(A, B, Q, R, controlState, ref, Kgains):
    """
    Controlled flight during hover phase, where the hopper maintains a constant altitude. Controller updates each 0.01s
    ------------------------------------------------------------------------------------------------
    A: Dynamics matrix of the system (anonymously defined)
    B: Control matrix of the system (anonymously defined)
    Q: State Weighting Matrix
    R: Control Weighting Matrix
    controlState: State Vector
    ref: Reference Setpoints
    Kgains: Controller Gains
    ------------------------------------------------------------------------------------------------
    """
    
    controller.stabilityCheck(A) # Check Stability
    controller.controllabilityCheck(A, B) # Check Controllability
    controller.observerabilityCheck(A, C) # Check Observability
    
    K, _, _ = ct.lqr(A, B, Q, R)  # LQR Controller to find optimal K for given Q and R
    
    # Control Output Calculation
    error = np.array([controlState[i] - ref[i] for i in range(len(controlState))])   
    u = -K @ error.T # NOTE: Anti-windup possible?     

    # Control Output Saturation
    u = np.clip(u, 0.3*maxThrust, maxThrust)

    # Update Control Input
    u0 = [u[0], u[1], u[2]]  

    # Storing K Gains
    Kgains = np.vstack((Kgains, K.flatten()))
    
    return u0, Kgains


#---------------- EXTERNAL DISTURBANCES -----------------#
def addNoise(x0, option=0):
    """
    This function adds noise to the state vector to simulate sensor noise.
        - Useful for robustness testing
        - NOTE: Maybe instead of statevector, do for each variable -> we can have different noise levels for each variable
    ------------------------------------------------------------------------------------------------
    x0: State Vector
    option: Type of Noise (0: White Noise, 1: Gaussian Noise)
    ------------------------------------------------------------------------------------------------    
    """
    # Define noise parameters
    match option:
        case 0:
            mean = 0
            std_dev = 0.4
            
        case 1:
            mean = 0
            std_dev = 0.4
    
    # Simulate noisy sensor data
    x0 += (0.001 * np.random.normal(mean, std_dev, len(x0)))
    return x0


#------------------- KALMAN FILTER -------------------#
f = KalmanFilter(dim_x=22, dim_z=22)


# Trajectory Formulation
ts = 15 # Simulation time (s)
dt = 0.0005 # Time step (s)
t = np.linspace(0, ts, int(ts/dt)) # Time Vector

if __name__ == "__main__":
    
#------------------- SIMULATION LOOP -------------------#
 
    result = x0 # Result Vector
    controls = u0 # Control Vector  
    earthTraj = [0, 0 ,0]
    gravity = [0, 0, -9.81]
    Kgains = np.zeros(rowA*colB) # Store K Gains
    
    for i in range(len(t)-1):
        
        temp = sc.integrate.solve_ivp(nlinsystem, (t[i], t[i+1]), x0, args=(u0,), method='RK45', t_eval=[t[i+1]]) # Integrate and solve
        x0 = temp.y[:,-1] # Update State Vector
        
        # External disturbances here
        x0[10:11] = addNoise(x0[10:11], 0)
        #x0[12:15] = addNoise(x0[12:15], 0)
        #
        #
        
        # Kalman filter update
        #
        #
        
        # Calculate trajectory relative to Earth-Fixed Axes -> For controller (i.e. Altitude control)
        i_earthTraj = earthTrajFunc(x0[0], x0[1], x0[2], x0[9], x0[10], x0[11])
        earthTraj = np.vstack((earthTraj, i_earthTraj.T)) 

        hoverControlState = np.concatenate((x0[2:3], x0[7:9], x0[10:12])) # Store control variables

        # Check if hopper has hit the ground, exit simulation if so
        if (i_earthTraj[2] >= 0):
            
            # Check if all fuel is depleted
            if (x0[12] >= m_dry):
                
                # Ascent Phase
                if (t[i] < 15):
                    u0, Kgains = flightControl(A(xEq, uEq), B(xEq, uEq), Q, R, hoverControlState, ref, Kgains)
                
                else:
                    print("What the fuck")
                    
                """  
                # Hover Phase
                elif (t[i] >= ascentTime and t[i] < hoverTime):
                    u0, Kgains = flightControl(A(x0, u0), B(x0, u0), Q, R, hoverControlState, ref, Kgains)

                    
                # Descent Phase
                elif (t[i] >= hoverTime and t[i] < hoverTime + descentTime):
                    u0, Kgains = flightControl(A(x0, u0), B(x0, u0), Q, R, hoverControlState, ref, Kgains)
                    
                """
                  
            else:
                pass
            
            result = np.vstack((result, x0)) # Store State Vector        
            controls = np.vstack((controls, u0)) # Store Control Inputs   
            
        else:
            print("Hopper has hit the ground.")
            break      
    
    # END FOR LOOP

#------------------- DATA CLEANING -------------------#
    #for i in [9, 10, 11]:
        #results.dataCleaning(result[:,i])

#------------------- PLOTS -------------------#

    results.plotResults(t, result, earthTraj)
    anim.flightAnimation(earthTraj, dt)
    

