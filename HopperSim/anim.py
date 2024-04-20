import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation

#------------------- FUNKY ANIMATION -------------------#

def getPlotSize(earthTraj, sizeMultiple=1.1):
    '''
    Finds max X, Y, or Z distance from the origin reached during the a flight. Used to set the 3D plot size (which will be equal in all dimensions)
    '''
    centerOfPlot = [np.mean(earthTraj[:, 0]), np.mean(earthTraj[:, 1]), np.mean(earthTraj[:, 2])]

    xRange = max(earthTraj[:, 0]) - min(earthTraj[:, 0])
    yRange = max(earthTraj[:, 1]) - min(earthTraj[:, 1])
    zRange = max(earthTraj[:, 2]) - min(earthTraj[:, 2])

    if max(xRange, yRange, zRange) == 0:
        axisDimensions = 1.0 # In the event of a stationary hopper
    else:
        axisDimensions = [xRange * sizeMultiple, yRange * sizeMultiple, zRange* sizeMultiple] # Ensure all axes are of equal length
    
    return axisDimensions, centerOfPlot   


def createFigure(axisDimensions, centreOfPlot):
    
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    
    ax.set_xlabel('x (m)')
    ax.set_ylabel('y (m)')
    ax.set_zlabel('z (m)')
    
    ax.set_xlim(centreOfPlot[0] - axisDimensions[0]/2, centreOfPlot[0] + axisDimensions[0]/2)
    ax.set_ylim(centreOfPlot[1] - axisDimensions[1]/2, centreOfPlot[1] + axisDimensions[1]/2)
    ax.set_zlim(0, centreOfPlot[2] + axisDimensions[2]/2)
    ax.set_title('Hopper Trajectory')
    
    return fig, ax


def flightAnimation(earthTraj, dt, hopperLength=0):
    
    # Create figure and 3D axis
    axisDimensions, centreOfPlot = getPlotSize(earthTraj, sizeMultiple = 1.1)
    fig, ax = createFigure(axisDimensions, centreOfPlot)
    
    # Using existing trajectory (earthTraj), compute where positions of COM (unchanged), engine and thrust vector are: maybe do this in main script and pass to function
    
    # animation
    line_traj, = ax.plot([], [], [], linewidth=2, color='gray')
    
    """
    xg, yg, zg, xh, yh, zh, xf, yf, zf = earthTraj[0, 3:]
    line_rocket, = ax.plot([yg, yh], [zg, zh], [xg, xh], linewidth=5, color='black')
    line_force, = ax.plot([yg, yf], [zg, zf], [xg, xf], linewidth=2, color='red')
    """
    """
    Objects needed: Rocket Body, Thrust Vector, Reference Trajectory
    I think i need to use absolute positions
    Attitude can be done by adding body fixed angles
    for now lets just try plotting the com
    """

    # Time label
    time_template = 'time = %.2fs'
    time_text = ax.text2D(0.66, 0.55, "time", transform=ax.transAxes)
    
    # initialization function: plot the background of each frame
    def init():
        line_traj.set_data([], [])
        line_traj.set_3d_properties([])
        return line_traj,

    # Animation function called SEQUENTIALLY
    def update_traj(i):
        time_text.set_text(time_template % (i * dt)) # Update time label
        line_traj.set_data(earthTraj[:i, 0], earthTraj[:i, 1]) # Update trajectory
        line_traj.set_3d_properties(earthTraj[:i, 2]) # Update trajectory
        return line_traj, time_text, # Return iterable artists (to be drawn)
    
    # call the animator.  blit=True means only re-draw the parts that have changed.
    ani = animation.FuncAnimation(fig, update_traj, init_func=init, frames=len(earthTraj), interval=1, blit=True)
    
    plt.show()
