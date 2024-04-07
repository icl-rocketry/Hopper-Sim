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


#------------------- SYMBOLIC VARIABLES -------------------#
x, y, z, xx, yy, zz, d, e, f, p, q, r, phi, theta, psi, m, Ixx, Iyy, Izz, xCG, yCG, zCG = symbols('x y z xx yy zz d e f p q r phi theta psi m Ixx Iyy Izz xCG yCG zCG')
Fpx, Fpy, Fpz = symbols('Fpx Fpy Fpz')


# Trajectory Formulation
ts = 13 # Simulation time (s)
dt = 0.0005 # Time step (s)
t = np.linspace(0, ts, int(ts/dt)) # Time Vector

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
6: d - Roll Angle in Body-Fixed Axes (rad) 
7: e - Pitch Angle in Body-Fixed Axes (rad) 
8: f - Yaw Angle in Body-Fixed Axes (rad)
9: p - Roll Rate (rad/s)
10: q - Pitch Rate (rad/s)
11: r - Yaw Rate (rad/s)
12: phi - Roll Euler Angle in Earth-Fixed Axes (rad)
13: theta - Pitch Euler Angle in Earth-Fixed Axes (rad) -> from Vertical
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
# CHECK INITIAL CONDITIONS

# ANGLES ARE WRONGLY DEFINED

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
x0[13] = np.pi/2
x0[14] = 0
x0[15] = m0
x0[16] = I_initial[0]
x0[17] = I_initial[1]
x0[18] = I_initial[2]
x0[19] = 1.2
x0[20] = 0
x0[21] = 0

"""
Control Vector:
0: Fpx - Force in body x-direction (N)
1: Fpy - Force in body y-direction (N)
2: Fpz - Force in body z-direction (N)
"""
u0 = np.zeros(3) 

# Initial Control (Force) Vector in BODY-FIXED Axes
u0 = [2500, 0, 0]


#------------------- ROTATIONAL MATRIX -------------------#

# https://www.intechopen.com/chapters/64567 
# https://society-of-flight-test-engineers.github.io/handbook-2013/axis-systems-and-transformations-raw.html
# https://www.aircraftflightmechanics.com/EoMs/EulerTransforms.html

# Earth to Body Transformation Matrix
#Teb = Matrix([[cos(theta)*cos(phi), cos(theta)*sin(phi), -sin(theta)],
#             [-cos(phi)*sin(psi) + sin(phi)*sin(theta)*cos(psi), cos(phi)*cos(psi) + sin(phi)*sin(theta)*sin(psi), sin(phi)*cos(theta)],
#             [sin(phi)*sin(psi) + cos(phi)*sin(theta)*cos(psi), -sin(phi)*cos(psi) + cos(phi)*sin(theta)*sin(psi), cos(phi)*cos(theta)]])

Teb = Matrix([[cos(theta)*cos(psi), sin(phi)*sin(theta)*cos(psi) - cos(phi)*sin(psi), cos(phi)*sin(theta)*cos(psi) + sin(phi)*sin(psi)],
             [cos(theta)*sin(psi), sin(phi)*sin(theta)*sin(psi) + cos(phi)*cos(psi), cos(phi)*sin(theta)*cos(psi) - sin(phi)*cos(psi)],
            [-sin(theta), sin(phi)*cos(theta), cos(phi)*cos(theta)]])
             
# By orthogonality, Tbe = Teb^-1
Tbe = Teb.T
Fgb = np.dot(Teb,Fge) # Gravity Vector in Body-Fixed Axes (N)

tebfunc = lambdify([phi, theta, psi], Teb, "scipy")

rocketCOM = [xCG, yCG, zCG] # Rocket Center of Mass (m), symoblically defined
momentArm = np.subtract(rocketCOM, engineCOM) # Moment Arm Vector (m)
moment = np.cross(momentArm, [Fpx, Fpy, Fpz]) # Moment Vector (N*m), symoblically defined

earthTrajMath = Matrix(np.dot(Tbe, [x, y, z])) # Trajectory in Earth-Fixed Axes
earthTrajFunc = lambdify([x, y, z, phi, theta, psi], earthTrajMath, "scipy")

# CHECK AXES DEFINITION IN DIFFERENT REFERENCE FRAMES

#------------------- STATE SPACE MODEL -------------------#

# For the dynamics of the system, we have a set of non-linear differential equations that describe the motion of the hopper. *dot* equations

stateFunction = Matrix([[xx],
            [yy],
            [zz],
            [Fpx/m + Fgb[0] - (q*zz - r*yy)], # Specific force in body x-direction
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
            [-sqrt(Fpx**2+Fpy**2+Fpz**2)/(Isp*g0)],
            [0],
            [0],
            [0],
            [-0.02],
            [0],
            [0]])

"""
Rate of change of CG and Inertias 
[((yCG - engineCOM[1])**2 + (zCG - engineCOM[2])**2) * -sqrt(Fpx**2+Fpy**2+Fpz**2)/(Isp*g0)],
            [((xCG - engineCOM[0])**2 + (zCG - engineCOM[2])**2) * -sqrt(Fpx**2+Fpy**2+Fpz**2)/(Isp*g0)],
            [((xCG - engineCOM[0])**2 + (yCG - engineCOM[1])**2) * -sqrt(Fpx**2+Fpy**2+Fpz**2)/(Isp*g0)],
            [-0.05 * xCG],
            [0],
            [0]
"""

# However, for control purposes, we need to linearise the system around a set of operating points.

controlFunction = Matrix([[(moment[1] - (r*p*(Ixx - Izz)))/Iyy],
            [(moment[2] - (p*q*(Iyy - Ixx)))/Izz]])

# Create ananonymous function for the non-linear system to be solved by the integrator
nln = lambdify(([(x, y, z, xx, yy, zz, d, e, f, p, q, r, phi, theta, psi, m, Ixx, Iyy, Izz, xCG, yCG, zCG), (Fpx, Fpy, Fpz)]), stateFunction, "scipy")

# Linearisation for LQR Control
jA = controlFunction.jacobian([q, r])
jB = controlFunction.jacobian([Fpx, Fpy, Fpz])

A = lambdify(([(x, y, z, xx, yy, zz, d, e, f, p, q, r, phi, theta, psi, m, Ixx, Iyy, Izz, xCG, yCG, zCG), (Fpx, Fpy, Fpz)]), jA, "scipy")
B = lambdify(([(x, y, z, xx, yy, zz, d, e, f, p, q, r, phi, theta, psi, m, Ixx, Iyy, Izz, xCG, yCG, zCG), (Fpx, Fpy, Fpz)]), jB, "scipy")

rowA, colA = jA.shape # Obtain dimensions of Jacobian Dynamics Matrix
rowB, colB = jB.shape # Obtain dimensions of Jacobian Control Matrix

C = np.diag([0, 0, 1, 
             0, 0, 0, 
             0, 0, 0, 
             0, 0, 0,
             0, 0, 0,
             0, 0, 0,
             0, 0, 0, 0]) # Control Matrix


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


def controllabilityCheck(A, B):    
#NOTE: Theta (pitch) is not controllable
    """
    This function checks the controllability of the system by checking the rank of the controllability matrix.
    ------------------------------------------------------------------------------------------------
    A: Dynamics matrix of the system (anonymously defined)
    B: Control matrix of the system (anonymously defined)
    ------------------------------------------------------------------------------------------------
    """
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


def observerabilityCheck(A, C):
    """
    This function checks the observability of the system by checking the rank of the observability matrix.
    ------------------------------------------------------------------------------------------------
    A: Dynamics matrix of the system (anonymously defined)
    B: Control matrix of the system (anonymously defined)
    ------------------------------------------------------------------------------------------------
    """   
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


def stabilityCheck(A):
    """
    NOTE: This is a stability check for LINEAR systems only -> Need to implement Lyapunov Stability for non-linear systems
    This function checks the stability of the system by checking the eigenvalues of the dynamics matrix.
    ------------------------------------------------------------------------------------------------
    A: Dynamics matrix of the system (anonymously defined)
    ------------------------------------------------------------------------------------------------
    """
    eig = np.linalg.eigvals(A)
    
    for i in eig:
        if i.real >= 0:
            print("System is unstable!")
            sys.exit()
            
    return 0



def dataCleaning(x):
    x *= 180/np.pi # Convert angles to degrees
    x[x >= 360] = x[x >= 360] - 360*(x[x >= 360] // 360) # Convert angles to 0-360 range



#---------------- CONTROLLER SETUP -----------------#

# Reference Setpoints in Earth-Fixed Frame
z_ref = 1 # Target Altitude: 1 m above ground, this needs to be in ECEF
q_ref = 0
r_ref = 0
theta_ref = np.pi/2
psi_ref = 0

ref = np.array([q_ref, r_ref])

# Use Bryson's rule to find optimal Q and R       
Q = np.diag([1, 1]) #NOTE: Maximum allowable value = 0 -> Q_i = 1
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


def ascentControl(x0):
    pass

def hoverControl():
    pass

def descentControl():
    pass

s = 0

#---------------- EXTERNAL DISTURBANCES -----------------#
def addNoise(x0, option):
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
            std_dev = 0.1
            
        case 1:
            mean = 0
            std_dev = 0.1
    
    # Simulate noisy sensor data
    noisyData = x0 + np.random.normal(mean, std_dev, len(x0))
    
    return noisyData


#------------------- KALMAN FILTER -------------------#
f = KalmanFilter(dim_x=22, dim_z=22)



if __name__ == "__main__":
    
#------------------- SIMULATION LOOP -------------------#
 
    result = x0 # Result Vector
    controls = u0 # Control Vector  
    earthTraj = [0, 0 ,0]
    gravity = [0, 0, -9.81]
    
    for i in range(len(t)-1):
        
        temp = sc.integrate.solve_ivp(nlinsystem, (t[i], t[i+1]), x0, args=(u0,), method='RK45', t_eval=[t[i+1]]) # Integrate and solve
        x0 = temp.y[:,-1] # Update State Vector
        
        # External disturbances here
        #
        #
        #
        
        # Kalman filter update
        #
        #
        
        # Calculate trajectory relative to Earth-Fixed Axes -> For controller (i.e. Altitude control)
        i_earthTraj = earthTrajFunc(x0[0], x0[1], x0[2], x0[12], x0[13], x0[14])
        earthTraj = np.vstack((earthTraj, i_earthTraj.T)) 
        gravity = np.vstack((gravity, np.dot(tebfunc(x0[10], x0[11], x0[12]), Fge))) # Gravity Vector in Body-Fixed Axes (N)

        #controlState = np.concatenate((x0[10:12], x0[13:15])) # Store control variables
        controlState = x0[11:13]
        
        # Check if hopper has hit the ground, exit simulation if so
        if (i_earthTraj[2] > 0):
            
            # Check if all fuel is depleted
            if (x0[15] >= m_dry):
                
                # Ascent Phase
                if (t[i] < ascentTime):
                    pass
                
                # Hover Phase
                elif (t[i] >= ascentTime and t[i] < hoverTime):
                    
                    stabilityCheck(A(x0, u0)) # Check Stability
                    controllabilityCheck(A(x0, u0), B(x0, u0)) # Check Controllability
                    #observerabilityCheck(A(x0, u0), C) # Check Observability
                    
                    K, _, _ = ct.lqr(A(x0, u0), B(x0, u0), Q, R)  # LQR Controller to find optimal K for given Q and R
                    
                    # Control Output Calculation
                    error = np.array([controlState[i] - ref[i] for i in range(len(controlState))])   
                    u = -K @ error.T # NOTE: Anti-windup possible?     

                    # Control Output Saturation
                    u = np.clip(u, 0.3*maxThrust, maxThrust)

                    # Update Control Input
                    u0 = [u[0], u[1], u[2]]  

                # Descent Phase
                elif (t[i] >= hoverTime and t[i] < hoverTime + descentTime):
                    descentControl()
                    
                else:
                    print("What the fuck")
                    break
                
            else:
                if (s == 0):
                    s = i # Store counter for fuel depletion
                if (outOfFuel == False):
                    u0 = thrustDecay(u0, 3, dt, s, i) # Simulate thrust decay
                else:
                    u0 = [0, 0 ,0]
            
            result = np.vstack((result, x0)) # Store State Vector        
            controls = np.vstack((controls, u0)) # Store Control Inputs   
            
            
        else:
            print("Hopper has hit the ground.")
            break      
    
    # END FOR LOOP

#------------------- DATA CLEANING -------------------#
    for i in [6,7,8,12,13,14]:
        dataCleaning(result[:,i])


#------------------- PLOTS -------------------#

    titles = ["x", "y", "z", "xx", "yy", "zz", "d", "e", "f", "p", "q", "r", "phi", "theta", "psi", "m", "Ixx", "Iyy", "Izz", "xCG", "yCG", "zCG"] # Titles for plots
    
    if (len(result) != len(t)):
        t = t[0:len(result)] # In the event the simulation ends early
    
    # 3D Trajectory
    plt.figure(1)
    ax = plt.axes(projection='3d')
    ax.plot3D(earthTraj[:,0], earthTraj[:,1], earthTraj[:,2], 'gray')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.set_title('ECEF Trajectory (?)')

    # Velocity Plots
    plt.figure(2)
    for i in range(3,6):
        plt.subplot(3, 1, i-2)
        plt.title(titles[i])
        plt.plot(t, result[:,i])
    
    # Earth Fixed Euler Angles
    plt.figure(3)
    for i in range(6, 9):
        plt.subplot(3, 1, i-5)
        plt.title(titles[i])
        plt.plot(t, result[:,i])
    
    # Angular Rates
    plt.figure(4)
    for i in range(9, 12):
        plt.subplot(3, 1, i-8)
        plt.title(titles[i])
        plt.plot(t, result[:,i])
    
    # Body Fixed Euler Angles
    plt.figure(5)
    for i in range(12, 15):
        plt.subplot(3, 1, i-11)
        plt.title(titles[i])
        plt.plot(t, result[:,i])
    
    # Inertia Tensor
    plt.figure(6)
    for i in range(16, 19):
        plt.subplot(3, 1, i-15)
        plt.title(titles[i])
        plt.plot(t, result[:,i])
    
    # Center of Gravity 
    plt.figure(7)
    for i in range(19, 22):
        plt.subplot(3, 1, i-18)
        plt.title(titles[i])
        plt.plot(t, result[:,i])
    
    # Mass
    plt.figure(8)
    plt.plot(t, result[:,15])
    plt.title("Mass Decay")
    
 
    titles2 = ["Fpx", "Fpy", "Fpz"]
 
    plt.figure(10)
    for i in range(0,3):
        plt.subplot(3, 1, i+1)
        plt.title(titles2[i])
        plt.plot(t, controls[:,i])


    plt.show()
    
    
#------------------- FUNKY ANIMATION -------------------#

def flightAnimation(flights, showPlot=True, saveAnimFileName=None):
    '''
    '''
    #### Set up data for animation ####
    # Filter out extra frames - seems like there's a limit to the number of frames that will work well with matplotlib, otherwise the end of the animation is weirdly sped up
    flights = _keepNTimeSteps(flights, nFramesToKeep=900)

    # Transform position info into arrays of x, y, z coordinates
    for flight in flights:
        Positions = [ [], [], [] ]
        for state in flight.rigidBodyStates:
            for i in range(3):
                Positions[i].append(state.position[i])
        flight.Positions = Positions

    # Set xyz size of plot - equal in all dimensions
    axisDimensions, centerOfPlot = _get3DPlotSize(flights[0]) # Assumes the top stage travels the furthest

    # Calculate frames at which engine turns off
    for flight in flights:
        flight.engineOffTimeStep = _findEventTimeStepNumber(flight, flight.engineOffTime)


    refAxis, perpVectors = _createReferenceVectors(nCanards, axisDimensions)

    #### Create Initial Plot ####
    fig, ax = _createAnimationFigure(axisDimensions, centerOfPlot)
    cgPoints, flightPathLines, longitudinalRocketLines, allCanardLines, allPerpLines = _createInitialFlightAnimationPlot(ax, nCanards, flights, refAxis, perpVectors)

    # Play animation
    ani = animation.FuncAnimation(fig, _update_plot, frames=(len(Positions[0]) - 1), fargs=(flights, refAxis, perpVectors, cgPoints, flightPathLines, longitudinalRocketLines, allCanardLines, allPerpLines), interval=1, blit=False, save_count=0, repeat_delay=5000)

    if showPlot:
        plt.show()

    if saveAnimFileName != None:
        ani.save("SampleRocket.mp4", bitrate=2500, fps=60)