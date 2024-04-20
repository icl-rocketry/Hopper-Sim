#TODO: plot results here, write big ass function to plot
import numpy as np
import matplotlib.pyplot as plt

def dataCleaning(x):
    x *= 180/np.pi # Convert angles to degrees
    x[x >= 360] = x[x >= 360] - 360*(x[x >= 360] // 360) # Convert angles to 0-360 range
    
def plotResults(t, result, earthTraj):
    
    titles = ["x", "y", "z", "xx", "yy", "zz", "p", "q", "r", "phi", "theta", "psi", "Ixx", "Iyy", "Izz"] # Titles for plots
    
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
    
    # Body-Fixed Angular Rates
    plt.figure(4)
    for i in range(6, 9):
        plt.subplot(3, 1, i-5)
        plt.title(titles[i])
        plt.plot(t, result[:,i])
    
    # Earth-Fixed Euler Angles
    plt.figure(5)
    for i in range(9, 12):
        plt.subplot(3, 1, i-8)
        plt.title(titles[i])
        plt.plot(t, result[:,i])
    
    # Inertia Tensor
    plt.figure(6)
    for i in range(13, 16):
        plt.subplot(3, 1, i-12)
        plt.title(titles[i-1])
        plt.plot(t, result[:,i])
    
    # Mass
    plt.figure(7)
    plt.plot(t, result[:,12])
    plt.title("Mass Decay")
    
    
    plt.show()
