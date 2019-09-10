from wilson import *
alphas=[0.1,0.05,.01]
p=.55
ps=np.arange(1.0,0.51,-0.001)
probval=[]
nval=[]
xval=[]
import matplotlib.pyplot as plt
import numpy as np
def update_line(hl, newx,newy):
    hl.set_xdata(np.append(hl.get_xdata(), newx))
    hl.set_ydata(np.append(hl.get_ydata(), newy))
    plt.draw()

import matplotlib.ticker as ticker
probvals=[]
nvals=[]
labels=[]
for alpha in alphas:
    labels.append(alpha)
    probvals.append([])
    nvals.append([])
    for p in ps:
        for n in range(1,1000):
            x=n*p
            x=int(x)
            if x/n>p :
                #This cannot be achieved with whole number so keep going
               continue

            l,u,mean=wils_int(x,n,alpha=alpha,twosided=True)
            if l>0.5:
                probvals[-1].append(p)
                nvals[-1].append(n)
                xval.append(x)
                print(f"{p}={x}/{n}")
                print(f"{l},{u}")
                print(f"{n} games needed for wilson L to predict p={p} to have L>0.5")
                break

def plotProbVn(probvals,nvals,labels=[]): #this was added without checking
    fig=plt.figure(figsize=(16.0, 16.0))
    ax = fig.add_subplot(1,1,1)
    minx=0.5
    ax.set_xlim(minx,1.0)
    major_xticks = np.arange(minx, 1, .1)
    ax.xaxis.set_ticks(major_xticks)
    minor_xticks = np.arange(minx, 1, .01)
    ax.xaxis.set_ticks(minor_xticks, minor=True)
    ax.grid(b=True, which='major',color='k', linestyle='-', alpha=0.9)
    ax.grid(True, which="minor",ls="--", alpha=0.5)

    start, end = ax.get_xlim()
    for idx,_ in enumerate(probvals):
        ax.semilogy(probvals[idx], nvals[idx])
    ax.set_xlabel("Probability of winning")
    ax.set_ylabel("Number of games")

    ax.legend(labels)

    plt.ylim(10**0)
    plt.savefig(f"MinSampleSizeFor_{1-alpha}_conf_stopping_LCB.png")
    plt.draw()
    plt.show()
plotProbVn(probvals,nvals,labels)