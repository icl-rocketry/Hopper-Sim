def _keepNTimeSteps(flights, nFramesToKeep=600):
    '''
        Pass in list of RocketFlight objects and the number of frames/timesteps you'd like to keep
        Will return a matching list of Rocket Flight objects with linearly-interpolated, evenly spaced (in time) timesteps
    '''
    unpackResult = False
    if isinstance(flights, RocketFlight):
        unpackResult = True
        flights = [ flights ]

    maxLength = len(flights[0].times)
    
    # Find the max flight time of any stage
    maxTime = 0
    for flight in flights:
        if flight.times[-1] > maxTime:
            maxTime = flight.times[-1]

    newTimes = np.linspace(0, maxTime, num=nFramesToKeep)

    for flight in flights:
        # Interpolate flights to the times in newTimes
        interpStates = []

        interpCanardDefls = None
        if flight.actuatorDefls != None:
            nCanards = len(flight.actuatorDefls)
            interpCanardDefls = [ [] for i in range(nCanards) ]

        for time in newTimes:
            # SmallYIndex, SmallYWeight, largeYIndex, largeYWeight (for linear interpolation)
            smY, smYW, lgY, lgYW = linInterpWeights(flight.times, time)

            # Interpolate rigid body state
            if type(flight.rigidBodyStates[smY]) == type(flight.rigidBodyStates[lgY]):
                interpolatedState = interpolateRigidBodyStates(flight.rigidBodyStates[smY], flight.rigidBodyStates[lgY], smYW)
            else:
                # Handles the switch from 6DoF to 3DoF, where two adjacent states will be of different types
                interpolatedState = flight.rigidBodyStates[lgY] if lgYW > smYW else flight.rigidBodyStates[smY]

            interpStates.append(interpolatedState)

            # Interpolate canard deflections
            if flight.actuatorDefls != None:
                for i in range(nCanards):
                    # Interpolate the deflection of each canard
                    interpolatedDeflection = flight.actuatorDefls[i][smY]*smYW + flight.actuatorDefls[i][lgY]*lgYW
                    interpCanardDefls[i].append(interpolatedDeflection)
        
        flight.times = newTimes
        flight.rigidBodyStates = interpStates
        if flight.actuatorDefls != None:
            flight.actuatorDefls = interpCanardDefls

    if unpackResult:
        return flights[0]
    else:
        return flights

def _get3DPlotSize(flight, sizeMultiple=1.1):
    '''
        Finds max X, Y, or Z distance from the origin reached during the a flight. Used to set the 3D plot size (which will be equal in all dimensions)
    '''
    Positions = flight.Positions
    centerOfPlot = Vector(mean(Positions[0]), mean(Positions[1]), mean(Positions[2]))

    xRange = max(Positions[0]) - min(Positions[0])
    yRange = max(Positions[1]) - min(Positions[1])
    zRange = max(Positions[2]) - min(Positions[2])

    if max(xRange, yRange, zRange) == 0:
        # For cases where the object does not move
        axisDimensions = 1.0
    else:
        axisDimensions = max([xRange, yRange, zRange]) * sizeMultiple
    
    return axisDimensions, centerOfPlot

def _findEventTimeStepNumber(flight, time):
    '''
        Given a time and a RocketFlight object, finds the time step that passes the given time
    '''
    if time == None:
        return None
    else:
        return bisect_right(flight.times, time) # Binary search for time value in 

def _createReferenceVectors(nCanards, maxAbsCoord, rocketLengthFactor=0.25, finLengthFactor=0.05):
    '''
        Creates a longitudinal vector and an array of nCanards perpendicular vectors. 
        These represent the rocket in the local frame and are rotated according to it's rigid body state at each time step.
        The size of the longitudinal and perpendicular lines are controlled by the rocketLength and finLength factor arguments and the maxAbsCoord.
    '''
    # Create vector reoresenting longitudinal axis in local frame
    refAxis = Vector( 0, 0, maxAbsCoord*rocketLengthFactor )
    
    # Create vectors going perpedicularly out, in the direction of each fin/canard in local frame
    radiansPerFin = radians(360 / nCanards)
    finToFinRotation = Quaternion(axisOfRotation=Vector(0,0,1), angle=radiansPerFin)
    perpVectors = [ Vector( maxAbsCoord*finLengthFactor, 0, 0 ) ]
    for i in range(nCanards-1):
        newVec = finToFinRotation.rotate(perpVectors[-1])
        perpVectors.append(newVec)

    return refAxis, perpVectors

def _createAnimationFigure(axisDimensions, centerOfPlot):
    fig = plt.figure()
    ax = p3.Axes3D(fig)

    halfDim = axisDimensions/2
    
    ax.set_xlim3d([centerOfPlot[0] - halfDim, centerOfPlot[0] + halfDim])
    ax.set_ylim3d([centerOfPlot[1] - halfDim, centerOfPlot[1] + halfDim])
    ax.set_zlim3d([centerOfPlot[2] - halfDim, centerOfPlot[2] + halfDim])
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')

    return fig, ax

def _createInitialFlightAnimationPlot_singleRocket(ax, nCanards, flight, refAxis, perpVectors):
    '''
        Called once for each rocket to set up the animation.
        Creates all the lines required for the rocket (longitudinal + perpendiculars + canards)
        These are then modified during each time step
    '''
    # Create flight path line
    Positions = flight.Positions
    flightPath = ax.plot(Positions[0][0:1], Positions[1][0:1], Positions[2][0:1])[0]
    if flight.engineOffTime != None:
        flightPath.set_color("red")
    else:
        flightPath.set_color("black")
    
    colors = ["blue", "red", "green", "purple", "brown", "grey", "peru", "skyblue", "pink", "darkblue", "darkgreen" ]

    # Create rocket/fin lines
    cg, rocketBase, tip, perpTips, canardTips, canardTails = _getRocketPoints(0, flight, refAxis, perpVectors)
    cgPoint = ax.scatter3D([ cg.X ], [ cg.Y ], [ cg.Z ])
    longitudinalLine = ax.plot([rocketBase.X, tip.X], [rocketBase.Y, tip.Y], [rocketBase.Z, tip.Z])[0]
    
    # Create fin and canard lines
    canardLines = []
    perpLines = []
    for i in range(nCanards):
        perpLines.append(ax.plot([rocketBase.X, perpTips[i].X], [rocketBase.Y, perpTips[i].Y], [rocketBase.Z, perpTips[i].Z])[0])
        perpLines[-1].set_color(colors[i % len(colors)])
        canardLines.append(ax.plot([canardTips[i].X, canardTails[i].X], [canardTips[i].Y, canardTails[i].Y], [canardTips[i].Z, canardTails[i].Z])[0])
        canardLines[-1].set_color(colors[i % len(colors)])

    # Plot target location if it exists
    target = flight.targetLocation
    if target != None:
        ax.scatter3D( [ target.X ], [ target.Y ], [ target.Z ])

    return cgPoint, flightPath, longitudinalLine, canardLines, perpLines

def _createInitialFlightAnimationPlot(ax, nCanards, flights, refAxis, perpVectors):
    '''
        Loops through all the rockets, calls the _singleRocket version of this function for each one.
    '''
    cgPoints = []
    flightPathLines = []
    longitudinalRocketLines = []
    allCanardLines = []
    allPerpLines = []
    for flight in flights:
        cgPoint, flightPath, longitudinalLine, canardLines, perpLines = _createInitialFlightAnimationPlot_singleRocket(ax, nCanards, flight, refAxis, perpVectors)
        cgPoints.append(cgPoint)
        flightPathLines.append(flightPath)
        longitudinalRocketLines.append(longitudinalLine)
        allCanardLines.append(canardLines)
        allPerpLines.append(perpLines)
    return cgPoints, flightPathLines, longitudinalRocketLines, allCanardLines, allPerpLines


def _getRocketPoints(timeStepNumber, flight, refAxis, perpVectors):
    '''
        For each rocket and time step, called to find the coordinates of the:
            cg, rocketBase, tip, perpTips, canardTips, canardTails
    '''
    nCanards = len(perpVectors)
    try:
        # Try plotting the actual rocket - using the orienation (only present in a 6DoF sim)
        cg = flight.rigidBodyStates[timeStepNumber].position
        currOrientation = flight.rigidBodyStates[timeStepNumber].orientation 

        axisVector = currOrientation.rotate(refAxis)
        perpVectors = [ currOrientation.rotate(x) for x in perpVectors ]
        
        # Assume rocket's CG is about halfway along it's axis
        rocketBase = flight.rigidBodyStates[timeStepNumber].position - axisVector*0.5        
        tip = rocketBase + axisVector
        perpTips = [ rocketBase + x for x in perpVectors ]

    except AttributeError:
        # Otherwise plot parachute for 3DoF simulation (3DoF only used for descent)
        rocketBase = flight.rigidBodyStates[timeStepNumber].position   
        cg = flight.rigidBodyStates[timeStepNumber].position        
        axisVector = refAxis*0.75
        tip = rocketBase - refAxis*0.2

        if flight.mainChuteDeployTime == None or timeStepNumber < flight.mainChuteTimeStep:
            chuteSize = 0.35
        else:
            chuteSize = 1.0

        perpTips = [ rocketBase + x*chuteSize + axisVector for x in perpVectors ]
        canardTips = perpTips

        parachuteTip = rocketBase + axisVector*1.2
        canardTails = [parachuteTip]*nCanards

    return cg, rocketBase, tip, perpTips, canardTips, canardTails

def _update_plot(timeStepNumber, flights, refAxis, perpVectors, cgPoints, flightPathLines, longitudinalRocketLines, allCanardLines, allPerpLines):
    ''' 
        Plot Update function - This gets called every time step of the simulation, updates the data for each point and line in the plot
    '''

    rocketColors = [ 'black', 'blue', 'fuchsia', 'olive', 'maroon', 'purple', 'sienna' ]

    for i in range(len(flights)):
        # Get important rocket coordinates at current time step
        cg, rocketBase, tip, perpTips, canardTips, canardTails = _getRocketPoints(timeStepNumber, flights[i], refAxis, perpVectors)
        nCanards = len(perpVectors)
        
        # Extract data for current stage to simplify code below
        flightPath = flightPathLines[i]
        Positions = flights[i].Positions
        engineOffTimeStep = flights[i].engineOffTimeStep
        cgPoint = cgPoints[i]
        longitudinalLine = longitudinalRocketLines[i]
        perpLines = allPerpLines[i]
        canardLines = allCanardLines[i]

        def setRocketColor(clr):
            flightPath.set_color(clr)
            cgPoint.set_color(clr)
            longitudinalLine.set_color(clr)
            for line in perpLines:
                line.set_color(clr)
            for line in canardLines:
                line.set_color(clr)

        if timeStepNumber == 0:
            clr = rocketColors[i % len(rocketColors)]
            setRocketColor(clr)
            flightPath.set_color('black')    

        # Update Flight path
        flightPath.set_data(Positions[0][:timeStepNumber], Positions[1][:timeStepNumber])
        flightPath.set_3d_properties(Positions[2][:timeStepNumber])
        if i > 0:
            if timeStepNumber < engineOffTimeStep:
                flightPath.set_color("red")
                flightPath.set_alpha(1)
            else:
                flightPath.set_color("gray")
                flightPath.set_alpha(0.5)
        elif engineOffTimeStep == None or timeStepNumber >= engineOffTimeStep:
            flightPath.set_color("black")
        else:
            flightPath.set_color("red")
        
        # Update rocket CG and main line
        cgPoint._offsets3d = ([ cg.X ], [ cg.Y ], [ cg.Z ])
        longitudinalLine.set_data([rocketBase.X, tip.X], [rocketBase.Y, tip.Y])
        longitudinalLine.set_3d_properties([rocketBase.Z, tip.Z])

        # Update fins and canards
        for c in range(nCanards):
            # Update tail fins / orientation indicators / parachute
            perpLines[c].set_data([rocketBase.X, perpTips[c].X], [rocketBase.Y, perpTips[c].Y])
            perpLines[c].set_3d_properties([rocketBase.Z, perpTips[c].Z])
            # Update canards / parachute
            canardLines[c].set_data([canardTips[c].X, canardTails[c].X], [canardTips[c].Y, canardTails[c].Y])
            canardLines[c].set_3d_properties([canardTips[c].Z, canardTails[c].Z])

        # Turn lines gray if landed
        if cg == flights[i].rigidBodyStates[-1].position:
            setRocketColor('gray')

def flightAnimation(flights, showPlot=True, saveAnimFileName=None):
    '''
        Pass in a list of RocketFlight object(s). Intended to contain a single RocketFlight object, or multiple if it was a staged flight (one object per stage)
        showPlot controls where the animation is shown or not
        saveAnimFileName should be a string file name/path that the animation should be saved to ex: "SampleRocketFlightAnimation.mp4"
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

    # Calculate frames at which engine turns off and main chute deploys
    for flight in flights:
        flight.engineOffTimeStep = _findEventTimeStepNumber(flight, flight.engineOffTime)
        flight.mainChuteTimeStep = _findEventTimeStepNumber(flight, flight.mainChuteDeployTime)

    if flights[0].actuatorDefls != None:
        # Assume all actuated systems are canards #TODO: This needs updating
        # Assuming canards always on the top stage
        nCanards = len(flights[0].actuatorDefls)
    else:
        nCanards = 4 # Create canard lines to reuse them for the parachute

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