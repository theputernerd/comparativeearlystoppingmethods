
from wilson import wils_int,wilson_z_score_0min,wilson_lcb,wilson_conf_delta,wilson_z_score_100min,wilson_z_score_30min,wilsonNOCC_z_score_30min,wilsonNOCC_z_score_100min
from wald import wald_int
from lib import *

from bayes import bayesian_U
maxNgames = 100  # if the detector hasn't made its mind up by this many games it becomes a type 2 error.



ngames = 350
from mpl_toolkits.mplot3d import axes3d



def get3dData(fn):
    del_data=[]
    lcb_data=[]
    for p2W in range(ngames,0,-1):
        predicted = False
        for p1W in range(0, ngames):
            p1L, p1U, mean = getLimits(p1W, p2W, fn)
            del_data.append([p1W, p2W, p1U - p1L])
            lcb_data.append([p1W, p2W, p1L])


    return del_data, lcb_data




def interpretXYZ(x,y,z,pts=1000):
    from scipy.interpolate import griddata
    import random
    if len(x) > 0:
        x, y,z = zip(*random.sample(list(zip(x, y,z)), int(len(x) / 1)))
    xi = np.linspace(min(x), max(x),pts)
    yi = np.linspace(min(y), max(y),pts)
    zi = griddata((x, y), z, (xi[None,:], yi[:,None]), method='cubic')
    return xi,yi,zi

import matplotlib.pyplot as plt


def threeD(fn,name):
    fig = plt.figure(figsize=plt.figaspect(0.5))
    from matplotlib import cm
    del_data, lcb_data=get3dData(fn)

    color='k'#'#87adde'
    #####################################LCB
    X = []
    Y = []
    Z = []
    zFloor=[]
    f=0.5
    for i in lcb_data:
        X.append(i[0])
        Y.append(i[1])
        Z.append(i[2])
        zFloor.append(f)

    x = np.array(X)
    y = np.array(Y)
    z = np.array(Z)
    zF=np.array(zFloor)
    xi, yi, zi = interpretXYZ(x, y, z,100)
    xfi, yfi, zfi = interpretXYZ(x, y, zF,100)
    ax = fig.add_subplot(1, 2, 1, projection='3d')
    # ax = fig.gca(projection='3d')
    cs1 = ax.contourf(xi, yi, zi, 500, linewidths=1,cmap=cm.jet)
    cset = ax.contourf(xi, yi, zi, 100, zdir='z', offset=f, linewidths=1,colors=color)
    cset = ax.contourf(xi, yi, zi, 100, zdir='z', offset=-.1, linewidths=.25,cmap=cm.jet)
    #cset = ax.contour(xi, yi, zi, 100, zdir='z', offset=0.5, linewidths=0.5)
    ax.invert_yaxis()
    #plt.colorbar(cs, ax=ax)
    ax.set_xlabel('p1Wins')
    # ax.set_xlim(0, maxNgames)
    ax.set_ylabel('p2Wins')
    # ax.set_ylim(0, maxNgames)
    ax.set_zlabel('LCB Value')
    #ax.set_zlim(0, 1)
    ax.set_title(f"(Test 1) - Lower confidence value")
    #####################################Delta UCB-LCB
    X=[]
    Y=[]
    Z=[]
    for i in del_data:

        X.append(i[0])
        Y.append(i[1])
        Z.append(i[2])
    x=np.array(X)
    y=np.array(Y)
    z=np.array(Z)

    xi, yi, zi=interpretXYZ(x,y,z)

    ax = fig.add_subplot(1, 2, 2, projection='3d')
    #ax = fig.gca(projection='3d')
    cs=ax.contourf(xi, yi, zi,500,linewidths=1,cmap=cm.jet)
    cset = ax.contourf(xi, yi, zi,100, zdir='z', offset=0.1,linewidths=1,colors=color)
    cset = ax.contour(xi, yi, zi,100, zdir='z', offset=0.4,linewidths=.25,cmap=cm.jet)
    plt.colorbar(cs1,ax=ax)
    ax.set_xlabel('p1Wins')
    #ax.set_xlim(0, maxNgames)
    ax.set_ylabel('p2Wins')
    #ax.set_ylim(0, maxNgames)
    ax.set_zlabel('|UCB-LCB|')
    ax.set_zlim(0.4, 0)
    ax.set_title(f"(Test 2) Difference between upper and lower limits")
    ax.invert_yaxis()

    ###################################################################
    fig.suptitle(f"{name} ")
    plt.savefig(f"{name}3d.eps", format='eps')
    plt.show(block=False)


threeD(bayesian_U,"Bayesian Updating")
print("Next Graph")
threeD(wils_int,"Wilson")
print("Next Graph")

print("Done - Waiting")

import time
while True:
    plt.pause(.0001)