from wilson import *
alpha=0.05
win=5
n=5
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
ls=[]
ws=[]
ns=[]

nMaxes=[10,50,100,150,200,1000]
for nMax in nMaxes:
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    for n in range(nMax):

        for win in range(n+1):
            print(win)
            l, u, m = wils_int(win, n, alpha=alpha, twosided=False)
            ls.append(l)
            ns.append(n)
            ws.append(win)


    ax.scatter(ns, ws, ls, c='r', marker='.')
    ax.set_title(f"Max N:{nMax}")
    ax.set_xlabel('N')
    ax.set_ylabel('Win')
    ax.set_zlabel('LCB')
    plt.show()
