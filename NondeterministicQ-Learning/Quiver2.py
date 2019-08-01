import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.ticker import AutoMinorLocator
from matplotlib.ticker import FixedLocator
from pylab import fill
import os
from matplotlib import rcParams, cycler
import pandas as pd
import csv


def export_dataset(xy, name):
    xs, ys = zip(*xy)
    X = [xs, ys]
    with open('reward-logs/Datasets/'+name+'.csv', 'w') as csvFile:
        writer = csv.writer(csvFile)
        writer.writerows(X)
    csvFile.close()

#def plotAllCosts(files, labels):
#    plt.clf
#    plt.subplot(1, 2, 1)
#    plotAllCostsSubplot(files, labels)
#
#    plt.subplot(1, 2, 2)
#    plt.xscale("log")
#    plotAllCostsSubplot(files, labels)
#
#    plt.savefig('LearningRates.png',dpi=400, bbox_inches = 'tight')
#

def plotAllCosts(files, labels):
#    files = []
#    for (dirpath, dirnames, filenames) in os.walk(os.getcwd()+'/reward-logs/Datasets/'):
#        files.extend(filenames)
#        break
#    files = [f for f in files if f[-4:] == '.csv']
    datasets = [ pd.read_csv( os.getcwd()+'/reward-logs/Datasets/'+f ) for f in files ]

    xss = [ [int(di) for di in d] for d in datasets]
    yss = [ [float(d[di]) for di in d] for d in datasets]

    plt.clf


    f, (ax1, ax2) = plt.subplots(1, 2, sharey=True)


    f.suptitle('Total Reward vs Iterations for each action selection method')
    l=labels
    cmap = plt.cm.PuRd_r
    rcParams['axes.prop_cycle'] = cycler(color=cmap(np.linspace(0, 1, len(l))))

    for i in range(len(l)):
        ax1.plot(xss[i], yss[i], color=cmap((i)*(0.6*1/len(l))), label=l[i])
    ax1.legend(loc='right center', fancybox=True, shadow=True)

    ax1.set_ylabel('Total Reward')
    ax1.set_xlabel('Iteration')

    for i in range(len(l)):
        ax2.plot(xss[i], yss[i], color=cmap((i)*(0.6*1/len(l))))
    ax2.set_xscale("log")
    ax2.set_xlabel('Log Iteration')

    plt.plot()
    plt.savefig('LearningRates.png',dpi=400)#, bbox_inches = 'tight')

def plotCostCts(f_list):
    plt.imshow(f_list)
    plt.show()

def plotRewardOverTime(xy, filename):
    plt.clf()
    x, y = zip(*xy)
    plt.plot(x, y)
    plt.xlabel("Iteration")
    plt.ylabel("Reward")
    plt.title("Total reward of policy at each iteration")
    plt.savefig(filename, dpi=200)

def plotPolicy(X, Y, U, V, M, w, h, **kwargs):#show and save default to false. title, colorbar-label, filename
    plt.clf()

    mask = np.logical_or(U != 0, V != 0)
    X = X[mask]
    Y = Y[mask]
    U = U[mask]
    V = V[mask]

    fig1, ax1 = plt.subplots()

    if 'title' in kwargs:
        ax1.set_title(kwargs['title'])

    # Make the arrows
    Q = ax1.quiver(X, Y, U, V, M, units='x', pivot='middle', width=0.05,
                    cmap="autumn_r", scale=1/0.8)

    # Shade and label goal cell
    ax1.text(w-1+0.2, h-1+0.4, 'Goal', color="g", fontsize=15)#, transform=ax1.transAxes)
    fill([w-1,w,w,w-1], [h-1,h-1,h,h], 'g', alpha=0.2, edgecolor='g')

    # Create colorbar
    cbar = ax1.figure.colorbar(Q, ax=ax1)#, **cbar_kw)
    t=""
    if "cbarlbl" in kwargs:
        t = kwargs["cbarlbl"] # label colorbar
    cbar.ax.set_ylabel(t, rotation=-90, va="bottom")

    # make grid lines on center
    plt.xticks([0.5+i for i in range(w)], [i for i in range(w)])
    plt.yticks([0.5+i for i in range(h)], [i for i in range(h)])
    plt.xlim(0,w)
    plt.ylim(0,h)
    # make grid
    minor_locator1 = AutoMinorLocator(2)
    minor_locator2 = FixedLocator([j for j in range(h)])
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

def makeUVMcontinuous(Qtable, w, h):# DOESN'T WORK YET
    x,y,u,v = [],[],[],[]
    for i in range(w):
        xrow, yrow, urow, vrow=[],[],[],[]
        for j in range(h):
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
    u[w-1, h-1]=0
    v[w-1, h-1]=0
    return x,y,u,v

def unpack(table):
    return table.q

def nothing(table):
    return table

def makeUVM(Qtable, w, h):# use the max Q value over each action for each state to define the direction
    x,y,u,v = [],[],[],[]
    f = nothing
    # print(type(list(Qtable.values())[0]))
    if type(list(Qtable.values())[0]) is not float: #type(Qtable.values[0])==tuple:
        # print("check")
        f = unpack
    for i in range(w):
        xrow, yrow, urow, vrow=[],[],[],[]
        for j in range(h):
            yrow.append(j+0.5)
            xrow.append(i+0.5)
            # if Qtable[((i,j),'right')].q > Qtable[((i,j),'up')].q:
            if f( Qtable[((i,j),'right')] ) > f( Qtable[((i,j),'up')] ):
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
    u[w-1, h-1]=0
    v[w-1, h-1]=0
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
