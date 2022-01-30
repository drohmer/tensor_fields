import numpy as np
import matplotlib.pyplot as plt

import PySimpleGUI as sg


def field(x,y,x0,phi0):

    sigma = 2
    
    
    A = np.exp(-((x-x0)*(x-x0)+y*y)/sigma/sigma)


    phi = np.arctan2(y,x-x0)+phi0
    A_phi = np.exp(1.0j*phi)

    F = A*A_phi

    return F


def display():

    ax.clear()
    fig_quiver = ax.quiver(x[1::sub,1::sub],y[1::sub,1::sub],f.real[1::sub,1::sub]/f_abs[1::sub,1::sub],f.imag[1::sub,1::sub]/f_abs[1::sub,1::sub],color='c', width=0.002)
    fig_contour = ax.contour(x,y,f_abs,[0.5], colors='b', linewidths =4)
    fig_imshow = ax.imshow(f_abs, extent=[-a,a,-a,a], cmap='Oranges')
    fig_stream = ax.streamplot(x,y,f.real,f.imag, density=2, linewidth=2*f_abs, arrowstyle="-",color='k')




N = 300
a = 8
u = np.linspace(-a,a,N)

x,y = np.meshgrid(u,u)

distances = np.linspace(0,3,4*4)

fig, ax = plt.subplots()




f1 = field(x,y,  1.0, np.pi/2)
f2 = field(x,y, -1.0, np.pi/2)

f = f1+f2

f_abs = abs(f)

sub = 10

display()
# fig_quiver = ax.quiver(x[1::sub,1::sub],y[1::sub,1::sub],f.real[1::sub,1::sub]/f_abs[1::sub,1::sub],f.imag[1::sub,1::sub]/f_abs[1::sub,1::sub])
# fig_contour = ax.contour(x,y,f_abs,[0.5])
# fig_imshow = ax.imshow(f_abs, extent=[-a,a,-a,a], cmap='Oranges')
#fig_stream = ax.streamplot(x,y,f.real,f.imag)

ax.axis('equal')



fig.set_size_inches(8,8)


plt.show(block=False)



layout = [
         [sg.Slider(key='Distance',
         range=(0,4),
         default_value=2,
         resolution=0.01,
         size=(20,15),
         orientation='horizontal',
         font=('Helvetica', 12),
         enable_events=True)],
         [sg.Slider(key='Angle',
         range=(-np.pi,np.pi),
         default_value=0,
         resolution=0.01,
         size=(20,15),
         orientation='horizontal',
         font=('Helvetica', 12),
         enable_events=True)]
         ]



# Create the Window
window = sg.Window('Window Title', layout, location=(0,0))

# Event Loop to process "events"
while True:             
    event, values = window.Read()

    d = values['Distance']
    theta = values['Angle']
    
    #print(d)
    f1 = field(x,y,  d, 0)
    f2 = field(x,y, -d, theta)
    f = f1+f2#+f1*f2#+np.sqrt(f1*f1+f2*f2+0j)
    #print(f1[50,50]*f1[50,50]+f2[50,50]*f2[50,50])
    f_abs = abs(f)


    # ax.quiver(x[1::sub,1::sub],y[1::sub,1::sub],f.real[1::sub,1::sub]/f_abs[1::sub,1::sub],f.imag[1::sub,1::sub]/f_abs[1::sub,1::sub])
    # ax.contour(x,y,f_abs,[0.5])
    
    

    display()




    plt.draw()


    if event in (None, 'Cancel'):
        break

window.Close()









