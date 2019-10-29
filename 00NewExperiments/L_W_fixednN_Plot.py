from wilson import *
alpha=0.05
win=5
n=5
#from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

nMaxes=[5,10,50,100,150,200,1000]
firstLCB=None
firstUCB=None
for nMax in nMaxes:
    ls = []
    ws = []
    ns = []
    us=[]
    fig = plt.figure()
    ax = fig.add_subplot(111)
    n=nMax
    lcbtriggered=False
    ucbtriggered=False
    for win in range(n+1):
        print(win)
        l, u, m = wils_int(win, n, alpha=alpha, twosided=False)
        if not lcbtriggered and l>0.50:
            firstLCB=[win,l]
            lcbtriggered=True
        if not ucbtriggered and u < 0.50:
            firstUCB = [win,u]
            #ucbtriggered=True I want the last one that triggered this

        ls.append(l)
        ns.append(n)
        ws.append(win)
        us.append(u)


    ax.scatter(ws, ls, c='r', marker='.')
    ax.scatter(ws, us, c='b', marker='.')
    offset=0.01*n
    arrow=dict(arrowstyle="->",facecolor='black')
    ax.annotate(f"({firstLCB[0]},{round(firstLCB[1],2)})",firstLCB,(firstLCB[0],firstLCB[1]-.2),
                arrowprops=arrow)
    ax.annotate(f"({firstUCB[0]},{round(firstUCB[1],2)})", firstUCB, (firstUCB[0], firstUCB[1] + .2),
                arrowprops=arrow)
    #ax.text(firstLCB[0],firstLCB[1]-offset,f"({firstLCB[0]},{round(firstLCB[1],2)})")
    #ax.text(firstUCB[0],firstUCB[1]-offset,f"({firstUCB[0]},{round(firstUCB[1],2)})")

    ax.legend(["LCB","UCB"])
    ax.grid()
    ax.set_title(f"Max N:{nMax}")
    ax.set_xlabel('Wins')
    ax.set_ylabel('LCB')
    plt.show()
    ax.cla()
    ax.clear()
    fig.clear()
    plt.close(fig)
    fig.clf()
