import numpy as np
import matplotlib.pyplot as plt

import PySimpleGUI as sg


def tensor_sphere(x0,y0, L1, L2, theta):


    sigma2 = 4^2
    Txx=np.zeros((N,N))
    Txy=np.zeros((N,N))
    Tyy=np.zeros((N,N))
    for ku in range(N):
        for kv in range(N):
            
            x = xx[ku,kv]
            y = yy[ku,kv]
            d2 = (x-x0)*(x-x0)+(y-y0)*(y-y0)

            f = np.exp(-d2/sigma2)

            e = np.matrix([[L1,0],[0,L2]])
            R = np.matrix([[np.cos(theta),-np.sin(theta)],[np.sin(theta),np.cos(theta)]])
            
            phi = 3*np.arctan2(x-x0,y-y0) + theta
            R = np.matrix([[np.cos(phi),-np.sin(phi)],[np.sin(phi),np.cos(phi)]])


            tensor = f*R*e*np.transpose(R)
            
            Txx[ku,kv] = tensor[0,0]
            Txy[ku,kv] = tensor[1,0]
            Tyy[ku,kv] = tensor[1,1]
    return (Txx,Txy,Tyy)



N = 200
a = 8
u = np.linspace(-a,a,N)

xx,yy = np.meshgrid(u,u)

Tid = np.matrix([[1,0],[0,1]])

T1xx,T1xy,T1yy = tensor_sphere(0.0,0.0, 1.0,0.1,0)
T2xx,T2xy,T2yy = tensor_sphere(-5.5,0.0, 1.0,0.1,np.pi/2)
T3xx,T3xy,T3yy = tensor_sphere(0.0,-5.5, 1.0,0.1,0)

# Txx = T1xx + T2xx + T3xx
# Txy = T1xy + T2xy + T3xy
# Tyy = T1yy + T2yy + T3yy
Txx = np.zeros((N,N))
Tyy = np.zeros((N,N))
Txy = np.zeros((N,N))
for ku in range(N):
    for kv in range(N):
        T1 = np.matrix([[T1xx[ku,kv],T1xy[ku,kv]],[T1xy[ku,kv],T1yy[ku,kv]]])
        T2 = np.matrix([[T2xx[ku,kv],T2xy[ku,kv]],[T2xy[ku,kv],T2yy[ku,kv]]])
        T3 = np.matrix([[T3xx[ku,kv],T3xy[ku,kv]],[T3xy[ku,kv],T3yy[ku,kv]]])
        T = (T1+T2+T3)

        Txx[ku,kv]=T[0,0]
        Txy[ku,kv]=T[1,0]
        Tyy[ku,kv]=T[1,1]




F_norm = np.zeros((N,N))
for ku in range(N):
    for kv in range(N):


        tens = np.matrix([[Txx[ku,kv],Txy[ku,kv]],[Txy[ku,kv],Tyy[ku,kv]]])
        w,v = np.linalg.eig(tens)

        F_norm[ku,kv] = np.sqrt(w[0]*w[0]+w[1]*w[1])



fig, ax = plt.subplots()
ax.axis('equal')
fig.set_size_inches(8,8)
#plt.show(block=False)

S = 4
for ku in range(0,N,S):
    for kv in range(0,N,S):
        tens = np.matrix([[Txx[ku,kv],Txy[ku,kv]],[Txy[ku,kv],Tyy[ku,kv]]])
        w,v = np.linalg.eig(tens)

        L = 0.3
        if w[0]>=w[1]:
            ax.plot([xx[ku,kv]-L*w[0]*v[0,0],xx[ku,kv]+L*w[0]*v[0,0]],[yy[ku,kv]-L*w[0]*v[0,1],yy[ku,kv]+L*w[0]*v[0,1]],'b')
            ax.plot([xx[ku,kv]-L*w[1]*v[1,0],xx[ku,kv]+L*w[1]*v[1,0]],[yy[ku,kv]-L*w[1]*v[1,1],yy[ku,kv]+L*w[1]*v[1,1]],'r')
        else:
            ax.plot([xx[ku,kv]-L*w[0]*v[0,0],xx[ku,kv]+L*w[0]*v[0,0]],[yy[ku,kv]-L*w[0]*v[0,1],yy[ku,kv]+L*w[0]*v[0,1]],'r')
            ax.plot([xx[ku,kv]-L*w[1]*v[1,0],xx[ku,kv]+L*w[1]*v[1,0]],[yy[ku,kv]-L*w[1]*v[1,1],yy[ku,kv]+L*w[1]*v[1,1]],'b')


fig_contour = ax.contour(xx,yy,F_norm,[0.5], colors='b', linewidths =4)
fig_imshow = ax.imshow(F_norm, extent=[-a,a,-a,a], cmap='Oranges', origin='lower')
plt.show()

