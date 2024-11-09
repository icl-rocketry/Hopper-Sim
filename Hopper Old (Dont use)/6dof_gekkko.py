# -*- coding: utf-8 -*-
"""
Created on Fri Mar  8 17:53:00 2024

@author: lvk23
"""

# -*- coding: utf-8 -*-
"""
2Dof gekko test
"""
from gekko import GEKKO
import numpy as np
import matplotlib.pyplot as plt

m = GEKKO(remote = False)
m.options.MAX_ITER = 1000

delta_T = .1
T = 10
m.time = np.linspace(0,T,int(T/delta_T))

#testing inputs
gamma1_data = [1*np.pi/180 for i in range(5)] + [0 for i in range(95)]
gamma2_data = [1*np.pi/180 for i in range(5)] + [0 for i in range(95)]
mdot_data = [1 for i in range(30)] + [0 for i in range(70)]

#constants
g = m.Const(9.81)
Vrel =  m.Const(-1500)
m0 =  m.Const(100)
#control values
#gamma1 = m.Param(value=gamma1_data)
#gamma2 = m.Param(value=gamma2_data)
mdot = m.Param(value=mdot_data)

gamma1 = .05
gamma2 = .00


# manipulted variables


#gamma1 = m.MV(value = 0., lb=-0.26179939, ub= 0.26179939)
#gamma1.STATUS = 1
#gamma1.DMAX =  0.017   # max change per time step  //rad/s

#gamma2 = m.MV(value = .1, lb=-0.26179939, ub= 0.26179939)
#gamma2.STATUS = 1
#gamma2.DMAX =  0.0017/delta_T   # max change per time step  //rad/s



#mdot = m.MV(value=.0, lb=0, ub=1.2)
#mdot.STATUS = 0
#mdot.DMAX =  0.005 # max change per time step // kg/s /s

#controlled variables

u = m.CV(value=0.)
u.STATUS = 0
m.options.CV_TYPE=2
u.SP = 0
u.TR_INIT = 0
u.TAU = 2 #time constant

#theta = m.CV(value=0.00001)
#u.STATUS = 1
#m.options.CV_TYPE=2
#u.SP = 0.01
#u.TR_INIT = 0
#u.TAU = 1 #time constant


#variables
#time
t = m.Param(value=m.time)
#earth pos
#xe= m.Var(value=0.)
#ye = m.Var(value= 0.)
#ze = m.Var(value=0.)
# body vels
#u = m.Var(value=0.)
v = m.Var(value=0.)
w = m.Var(value=0.)
#mass
mass = m.Var(value=m0)



#pos on body
x =  m.Var(0)
y =  m.Var(0)
z =  m.Var(0)

#pos in earth 
xe =  m.Var(0)
ye =  m.Var(0)
ze =  m.Var(0)

#euler
theta = m.Var(value=0.00001)
phi =  m.Var(value=np.pi/2)
psi = m.Var(value=np.pi/2)

#body roll rates
q = m.Var(value=0)
p = m.Var(value=0)
r = m.Var(value=0)

# distance from engine zo CoM
dx = m.Var(value=1.5)


#inertias
Ixx = m.Var(value = 300000) 
Iyy = m.Var(value = 100000)
Izz = m.Var(value = 100000)


#####################################################################


#setting up forces and other relations
#gravity in body
Fgx = m.Intermediate(-1*g*mass*(m.sin(phi)*m.sin(psi)+m.sin(theta)*m.sin(phi)*m.sin(psi)))
Fgy = m.Intermediate(-1*g*mass*(m.sin(theta)*m.cos(psi)*m.cos(psi)-m.sin(phi)*m.cos(psi)))
Fgz = m.Intermediate(-1*g*mass*m.cos(phi)*m.cos(theta))

#Thrust in Body
Fx = m.Intermediate(Vrel*mdot*(m.cos(gamma1)*m.cos(gamma2)))
Fy = m.Intermediate(Vrel*mdot*(m.cos(gamma2)*m.sin(gamma1)))
Fz = m.Intermediate(-1*Vrel*mdot*(m.sin(gamma2)))

#Moments in Body
#My = m.Intermediate(Fz*dx)
#Mz = m.Intermediate(-Fy*dx)


#mass mdot relation
m.Equation(mass.dt() == -mdot)
#modelling change in inertia
m.Equation(mass.dt()*600 == Ixx.dt())
m.Equation(mass.dt()*600 == Iyy.dt())
m.Equation(mass.dt()*600 == Izz.dt())
#modelling change in distance from CoM to engine 
m.Equation(dx.dt() ==0.01*mass.dt())



#equations of motion

#accelerations
m.Equation(u.dt() == (Fgx - Fx)/mass - q*w + r*v)
m.Equation(v.dt() == (Fgy - Fy)/mass + p*w - r*u)
m.Equation(w.dt() == (Fgz - Fz)/mass - p*v + q*u)


m.Equation(p.dt() ==(-Iyy*q*r + Izz*q*r - p*Ixx.dt())/Ixx)
m.Equation(q.dt() ==(Fz*dx -Ixx*p*r + p*r*Iyy -q*Iyy.dt())/Izz)
m.Equation(r.dt() ==(-Fy*dx + Ixx*p*q -Izz*p*q -Iyy.dt()*r)/Iyy)



# euler second attempt
m.Equation(theta.dt() == p*m.cos(phi)*m.cos(theta) + q*m.sin(phi)*m.cos(theta)-r*m.sin(theta))
m.Equation(phi.dt() == q*(m.sin(phi)*m.sin(psi)*m.sin(theta) + m.cos(phi)*m.cos(psi)) + p*(m.sin(phi)*m.sin(theta)*m.cos(psi)-m.sin(psi)*m.cos(phi)) + r*(m.sin(phi)*m.cos(theta)))
m.Equation(psi.dt() == p*(m.sin(phi)*m.sin(psi)+m.sin(theta)*m.cos(phi)*m.cos(psi)) + q*(-m.sin(phi)*m.cos(psi)+m.sin(theta)*m.cos(phi)*m.cos(psi)) + r*(m.cos(phi)*m.cos(theta)))



#coordinates
m.Equation(x.dt() == u)
m.Equation(y.dt() == v)
m.Equation(z.dt() == w)

m.Equation(xe == x*m.cos(phi)*m.cos(theta) + y*m.sin(phi)*m.cos(theta)-z*m.sin(theta))
m.Equation(ye == z*(m.sin(phi)*m.sin(psi)*m.sin(theta) + m.cos(phi)*m.cos(psi)) + x*(m.sin(phi)*m.sin(theta)*m.cos(psi)-m.sin(psi)*m.cos(phi)) + z*(m.sin(phi)*m.cos(theta)))
m.Equation(ze == x*(m.sin(phi)*m.sin(psi)+m.sin(theta)*m.cos(phi)*m.cos(psi)) + y*(-m.sin(phi)*m.cos(psi)+m.sin(theta)*m.cos(phi)*m.cos(psi)) + z*(m.cos(phi)*m.cos(theta)))


m.options.IMODE = 4
m.solve()


plt.plot(m.time,u.value)
plt.figure()
plt.plot(m.time,xe.value, label="X")
plt.plot(m.time,ye.value, label="Y")
plt.plot(m.time,ze.value, label="Z")
plt.legend()

fig = plt.figure()
ax = plt.axes(projection="3d")
ax.plot3D(xe.value, ye.value, ze.value)
ax.set_zlim(0, max(ze.value)*1.1)

plt.show()
