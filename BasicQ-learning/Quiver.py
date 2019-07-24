import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.ticker import AutoMinorLocator
from matplotlib.ticker import FixedLocator
from pylab import fill


def plotPolicy(X, Y, U, V, M, **kwargs):#show and save default to false. title, colorbar-label, filename
    plt.clf()

    fig1, ax1 = plt.subplots()

    if 'title' in kwargs:
        ax1.set_title(kwargs['title'])

    # Make the arrows
    Q = ax1.quiver(X, Y, U, V, M, units='x', pivot='middle', width=0.05,
                    cmap="autumn_r", scale=1/0.8)

    # Shade and label goal cell
    ax1.text(4.2, 4.4, 'Goal', color="g", fontsize=15)#, transform=ax1.transAxes)
    fill([4,5,5,4], [4,4,5,5], 'g', alpha=0.2, edgecolor='g')

    # Create colorbar
    cbar = ax1.figure.colorbar(Q, ax=ax1)#, **cbar_kw)
    t=""
    if "cbarlbl" in kwargs:
        t = kwargs["cbarlbl"] # label colorbar
    cbar.ax.set_ylabel(t, rotation=-90, va="bottom")

    # make grid lines on center
    plt.xticks([0.5+i for i in range(5)], [i for i in range(5)])
    plt.yticks([0.5+i for i in range(5)], [i for i in range(5)])
    plt.xlim(0,5)
    plt.ylim(0,5)
    # make grid
    minor_locator1 = AutoMinorLocator(2)
    minor_locator2 = FixedLocator([i for i in range(5)])
    plt.gca().xaxis.set_minor_locator(minor_locator1)
    plt.gca().yaxis.set_minor_locator(minor_locator2)
    plt.grid(which='minor')

    plt.xlabel("cell x coord.")
    plt.ylabel("cell y coord.")

    if 'filename' in kwargs:
        plt.savefig(kwargs['filename'], dpi=200)
    if 'show' in kwargs:
        if kwargs['show']:
            plt.show()

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def makeUVMcontinuous(Qtable):# DOESN'T WORK YET
    x,y,u,v = [],[],[],[]
    for i in range(5):
        xrow, yrow, urow, vrow=[],[],[],[]
        for j in range(5):
            yrow.append(i+0.5)
            xrow.append(j+0.5)
            urow.append(Qtable[((i,j),'right')])
            vrow.append(Qtable[((i,j),'up')])
        x.append(xrow)
        y.append(yrow)
        u.append(urow)
        v.append(vrow)
    x = np.array(x)
    y = np.array(y)
    u = np.array(u)
    v = np.array(v)

    mnu = np.amin(u)
    mnv = np.amin(v)
    u-=mnu
    v-=mnv

    angles = arrAtan(v, u)
    h = np.hypot(u, v)
    mx = np.amax(h)
    if mx == 0:
        return x,y,u,v
    h = 8*h/mx - 4
    h = sigmoid(h)
    # print(h)
    u = h*np.cos(angles)
    v = h*np.sin(angles)
    u[4, 4]=0
    v[4, 4]=0
    return x,y,u,v

def makeUVM(Qtable):# use the max Q value over each action for each state to define the direction
    x,y,u,v = [],[],[],[]
    for i in range(5):
        xrow, yrow, urow, vrow=[],[],[],[]
        for j in range(5):
            yrow.append(i+0.5)
            xrow.append(j+0.5)
            if Qtable[((i,j),'right')] > Qtable[((i,j),'up')]:
                urow.append(1)
                vrow.append(0)
            else:
                urow.append(0)
                vrow.append(1)

        x.append(xrow)
        y.append(yrow)
        u.append(urow)
        v.append(vrow)
    x = np.array(x)
    y = np.array(y)
    u = np.array(u)
    v = np.array(v)
    u[4, 4]=0
    v[4, 4]=0
    return x,y,u,v

def atan(a, b):# NOT IN USE YET- helper fn for continuousUMV
    if b == 0:
        return np.pi/2
    return np.arctan(a/b)

def arrAtan(A, B):# NOT IN USE YET- helper fn for continuousUMV
    out = np.zeros(shape=(len(A),len(A[0])))
    for i in range(len(A)):
        for j in range(len(A[0])):
            out[i, j] = atan(A[i, j], B[i, j])
    return out
