# Packages Required: numpy, scipy, sympy 
import numpy as np
import matplotlib.pyplot as plt
from sympy import *
import scipy as sc

# Symbol definition
x, y, z, xx, yy, zz, d, e, f, p, q, r, phi, theta, psi = symbols('x y z xx yy zz d e f p q r phi theta psi')

# Function Definition
def RK4(f, x0, dt):
  """
  Runge-Kutta 4th Order Method
  Arguments:
      f  - function to be integrated
      tspan - time span [ti, tf]
      x0  - initial state
      dt  - time step
  Returns:
      newState - new state vector
  """
  k1 = dt * np.dot(f(*x0), x0) 
  k2 = dt * np.dot(f(*x0 + k1/2), x0)
  k3 = dt * np.dot(f(*x0 + k2/2), x0)
  k4 = dt * np.dot(f(*x0 + k3), x0)
  newState = x0 + (k1 + 2*k2 + 2*k3 + k4)/6
  return newState


# Earth to Body-Fixed Frame Transformation Matrix
Tbe = Matrix([[cos(theta)*cos(phi), sin(phi)*sin(theta)*cos(phi) - cos(phi)*sin(psi), cos(phi*sin(theta)*cos(psi) + sin(phi)*sin(psi))],
             [cos(theta)*sin(phi), sin(phi)*sin(theta)*sin(phi) + cos(phi)*sin(psi), sin(phi)*sin(theta)*cos(psi) - cos(phi)*sin(psi)],
             [-sin(theta), sin(phi)*cos(theta), cos(phi)*cos(theta)]])


# Trajectory Formulation
ts = 10 # Simulation time (s)
dt = 0.1 # Time step (s)
maxThrust = 1800 # Maximum Hopper Engine Thrust (N)

m0 = 125 # Initial Mass (kg)
Fp = [1800, 1800, 1800] # Thrust Vector (N)
Fge = [0, 0, -9.81] # Gravity Vector in Earth-Fixed Axes (m/s^2)
I = [27.920, 27.960, 7.87870] # Inertia Tensor (kg*m^2)
Isp = 190 # Specific Impulse (s)
rocketCOM = [5, 0, 0] # Rocket Center of Mass (m)
engineCOM = [10, 0, 0] # Engine Center of Mass (m)


momentArm = np.subtract(rocketCOM, engineCOM) # Moment Arm Vector (m)
moment = np.cross(momentArm, Fp) # Moment Vector (N*m)

# Initialise state vector
stateVector = np.zeros((int(ts/dt), 15)) # State Vector

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
"""

# Initial Conditions
stateVector[0,0] = 1
stateVector[0,1] = 1
stateVector[0,2] = 1
stateVector[0,3] = 1
stateVector[0,4] = 20
stateVector[0,5] = 1
stateVector[0,6] = 1
stateVector[0,7] = 1
stateVector[0,8] = 1
stateVector[0,9] = 1
stateVector[0,10] = 1
stateVector[0,11] = 1
stateVector[0,12] = 1
stateVector[0,13] = 1
stateVector[0,14] = 1

# Update Gravity    
Fgb = np.dot(Tbe, Fge) # Gravity Vector in Body-Fixed Axes (N)

# State Space Formulation
stateFunction = Matrix([[xx],
            [yy],
            [zz],
            [(Fp[0]+Fgb[0])/m0 - (q*zz - r*yy)],
            [(Fp[1]+Fgb[1])/m0 - (r*xx - p*zz)],
            [(Fp[2]+Fgb[2])/m0 - (p*yy - q*xx)],
            [p],
            [q],
            [r],
            [(moment[0] - (q*r*(I[1] - I[2])))/I[0]],
            [(moment[1] - (p*r*(I[2] - I[0])))/I[1]],
            [(moment[2] - (p*q*(I[0] - I[1])))/I[2]],
            [p + (q*sin(phi) + r*cos(phi))*tan(theta)],
            [q*cos(phi) - r*sin(phi)],
            [(q*sin(phi) + r*cos(phi))/cos(theta)]])

# Linearisation
JacobianState = stateFunction.jacobian([x, y, z, xx, yy, zz, d, e, f, p, q, r, phi, theta, psi]) # Note this should not be iterated in a loop, it should be done once only
f = lambdify(([x, y, z, xx, yy, zz, d, e, f, p, q, r, phi, theta, psi]), JacobianState, "scipy") # Convert to a function -> scipy array
 
if __name__ == "__main__":
    for ii in range(int(ts/dt)-1):
        m0 -= (maxThrust)/Isp*dt # Mass Update , I was thinking if we have mass as a state variable, we can update it in the state function instead of having to do it here   
        stateVector[ii+1,:] = RK4(f, stateVector[ii,:], dt) # Solve for each time step
    
    # Plotting
    time = np.linspace(0, ts, int(ts/dt))
    
    titles = ['x', 'y', 'z', 'xx', 'yy', 'zz', 'd', 'e', 'f', 'p', 'q', 'r', 'phi', 'theta', 'psi']
    
    fig1 = plt.figure(1)
    fig1.suptitle('States')
    for ii in range(15):
        a = plt.subplot(5,3,ii+1)
        a.set_title(titles[ii])
        plt.plot(time, stateVector[:,ii])   
    
    plt.show()
    

    
