from lib import shouldIStop
from lib import bayesianU_int,wils_int
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

import numpy as np
import matplotlib.ticker
#####################################################
def cond1Stop_probX_nY(fn,ngames):
    W1x = []
    W1y = []
    W2x = []
    W2y = []
    probs=[]
    minX=0
    for p2w in range(0, ngames):
        stop = False
        WStop1 = False

        for p1w in range(0, ngames):
            if p1w < minX:
                continue
            n = p1w + p2w

            ###Cond1
            L, U, mean = fn(p1w, n, 0.05)
            shouldI, type = shouldIStop(1, L, U, mean, n, delta=0.1)
            if shouldI and type == 1 and not WStop1:
                p=p1w/n
                W1x.append(p)
                W1y.append(n)
                probs.append(p)
                WStop1 = True

            if WStop1:
                minX=p1w
                stop = True
                break
        if stop:
            stop = False
            continue
    return W1x,W1y
def cond2Stop_probX_nY(fn,ngames):
    W1x = []
    W1y = []
    W2x = []
    W2y = []
    probs=[]
    minX=0
    for p2w in range(0, ngames):
        stop = False
        WStop1 = False

        for p1w in range(0, ngames):
            if p1w < minX:
                continue
            n = p1w + p2w

            ###Cond1
            L, U, mean = fn(p1w, n, 0.05)

            shouldI1, type = shouldIStop(1, L, U, mean, n, delta=0.1)
            if shouldI1:
                pass
                #continue
            L, U, mean = fn(p1w, n, 0.025)
            shouldI, type = shouldIStop(3, L, U, mean, n, delta=0.1)
            if shouldI and not WStop1:

                p=p1w/n
                if p not in probs:
                    W1x.append(p)
                    W1y.append(n)
                    probs.append(p)
                    WStop1 = True

            if WStop1:
                minX=p1w
                stop = True
                break
        if stop:
            stop = False
            continue
    return W1x,W1y
show=False
def cameronsPlot(ngames):
    Bx = []
    By = []


    W2x, W2y = cond2Stop_probX_nY(wils_int, ngames)
    W1x, W1y = cond1Stop_probX_nY(wils_int, ngames)

    name = "wilsonStopping"
    name=f"{ngames}_"+name

    fig, ax = plt.subplots(1, 1, figsize=(19.20, 10.8))
    gcolor = '#b7b7bc'
    ax.grid(color=gcolor, linestyle='-', linewidth=1)
    plt.grid(b=True, which='minor', color=gcolor, linestyle='-', alpha=0.5)
    plt.minorticks_on()
    WPlot, = ax.plot(W1x, W1y, 'bo', label="wilson_cond1", markersize=1)
    wDraws, = ax.plot(W2x, W2y, 'go', label="wilson_Cond2", markersize=1)
    plt.legend(handles=[WPlot, wDraws])
    ax.set_ylim(ymin=0)
    ax.set_xlim(xmin=0)
    plt.xlabel("Probability")
    plt.ylabel("Ngames")
    plt.title(f"{name}")
    plt.grid(b=True, which='minor', color=gcolor, linestyle='-', alpha=0.5)
    plt.savefig(f"{name}.pdf", format='pdf')
    if show:
        plt.show()
    plt.close()
###################################################
def cond1Stop_X_nY(fn,ngames):
    W1x = []
    W1y = []
    W2x = []
    W2y = []
    probs=[]
    minX=0
    for p2w in range(0, ngames):
        stop = False
        WStop1 = False

        for p1w in range(0, ngames):
            #if p1w < minX:
            #    continue
            n = p1w + p2w

            ###Cond1
            L, U, mean = fn(p1w, n, 0.05)
            shouldI, type = shouldIStop(1, L, U, mean, n, delta=0.1)
            if shouldI  and not WStop1:
                p=p1w/n
                W1x.append(p)
                W1y.append(n)
                probs.append(p)
                #WStop1 = True

            if WStop1:
                minX=p1w
                stop = True
                break
        if stop:
            stop = False
            continue
    return W1x,W1y
def cond2Stop_P1X_nY(fn,ngames):
    W1x = []
    W1y = []
    W2x = []
    W2y = []
    probs=[]
    minX=0
    for p2w in range(0, ngames):
        stop = False
        WStop1 = False

        for p1w in range(0, ngames):
            if p1w < minX:
                continue
            n = p1w + p2w

            ###Cond1
            L, U, mean = fn(p1w, n, 0.05)

            shouldI1, type = shouldIStop(1, L, U, mean, n, delta=0.1)
            if shouldI1:
                pass
                #continue
            L, U, mean = fn(p1w, n, 0.025)
            shouldI, type = shouldIStop(3, L, U, mean, n, delta=0.1)
            if shouldI and not WStop1:

                p=p1w/n
                #if p not in probs:
                W1x.append(p)
                W1y.append(n)
                probs.append(p)
                #WStop1 = True

            if WStop1:
                minX=p1w
                stop = True
                break
        if stop:
            stop = False
            continue
    return W1x,W1y
def cond3Stop_P1X_nY(fn,ngames):
    W1x = []
    W1y = []
    W2x = []
    W2y = []
    probs=[]
    minX=0
    for p2w in range(0, ngames):
        stop = False
        WStop1 = False

        for p1w in range(0, ngames):
            if p1w < minX:
                continue
            n = p1w + p2w

            ###Cond1

            L, U, mean = fn(p1w, n, 0.05)
            shouldI1, type = shouldIStop(1, L, U, mean, n, delta=0.1)
            if shouldI1:
                continue
            L, U, mean = fn(p1w, n, 0.025)
            shouldI, type = shouldIStop(3, L, U, mean, n, delta=0.1)
            if shouldI and not WStop1:

                p=p1w/n
                #if p not in probs:
                W1x.append(p)
                W1y.append(n)
                probs.append(p)
                #WStop1 = True

            if WStop1:
                minX=p1w
                stop = True
                break
        if stop:
            stop = False
            continue
    return W1x,W1y
####################################################
#load csv data if exists or collects data and saves it to csv. Awins,Bwins where shouldIstop is True
def cond1Stop_p1w_p2w(fn,ngames,alpha=0.05,delta=0.05,newData=False):

    #returns x,y data of p1wins vs p2wins using condition 1 stopping for a max number of games
    read=False
    filename = f"data/{ngames}_{alpha}_{delta}_{fn.__name__}_C1.csv"
    W1x = []
    W1y = []
    import os
    if not newData and os.path.exists(filename): #then read the file so I dont have to generate again
        print(f"Loading Data for C1 {filename}")
        import csv
        with open(filename, 'r') as csvfile:
            spamreader=csv.reader(csvfile,delimiter=',')
            for row in spamreader:
                if row==[]: continue
                x,y = row
                W1x.append(int(x))
                W1y.append(int(y))
        read==True #don't save it, cause I didnt create anything.
    else: #have to create the data
        print(f"Creating new Data for C1 {filename}")
        for p2w in range(0, ngames):
            for p1w in range(0, ngames):
                n = p1w + p2w
                ###Cond1
                L, U, mean = fn(p1w, n, alpha)
                shouldI, type = shouldIStop(1, L, U, mean, n, delta=delta)

                if shouldI :
                    W1x.append(p1w)
                    W1y.append(n-p1w)


    ##write to file so I don't have to keep re-running it.
    if not read: #then write the file so I dont have to generate again
        import csv
        with open(filename, 'w', newline='') as csvfile:
            spamwriter=csv.writer(csvfile,delimiter=',')
            xyval=zip(W1x,W1y)
            for xy in xyval :
                x,y=[*zip(xy)]
                spamwriter.writerow(list(xy))
        print(f"Data written {filename}")

    return W1x,W1y
def oldC2cond3Stop_p1w_p2w(fn,ngames,alpha=0.05,delta=0.05,newData=False):

    #returns x,y data of p1wins vs p2wins using condition 1 stopping for a max number of games
    read=False
    filename = f"data/{ngames}_{alpha}_{delta}_{fn.__name__}_C2old.csv"
    W1x = []

    W1y = []
    W2x = []
    W2y = []
    probs=[]
    minX=0
    import os
    if not newData and os.path.exists(filename): #then read the file so I dont have to generate again
        print(f"Loading Data for C2 {filename}")
        import csv
        with open(filename, 'r') as csvfile:
            spamreader=csv.reader(csvfile,delimiter=',')
            for row in spamreader:
                if row==[]: continue
                x,y = row
                W1x.append(int(x))
                W1y.append(int(y))
        read=True #don't save it, cause I didnt create anything.
    else:
        print(f"Creating new Data for C2 {filename}")
        for p2w in range(0, ngames):

            for p1w in range(0, ngames):
                if p1w < minX:
                    continue
                n = p1w + p2w

                ###Cond1
                L, U, mean = fn(p1w, n, alpha)
                shouldI1, type = shouldIStop(1, L, U, mean, n, delta=delta)
                if shouldI1:
                    pass #type 1 wouldve stopped this so dont check next.
                    #continue
                L, U, mean = fn(p1w, n, alpha/2.0)
                shouldI, type = shouldIStop(3, L, U, mean, n, delta=delta)
                if shouldI :

                    W1x.append(p1w)
                    W1y.append(n-p1w)


    if not read: #then write the file so I dont have to generate again
        import csv
        with open(filename, 'w', newline='') as csvfile:
            spamwriter=csv.writer(csvfile,delimiter=',')
            xyval=zip(W1x,W1y)
            for xy in xyval :
                x,y=[*zip(xy)]
                spamwriter.writerow(list(xy))
        print(f"Data written {filename}")

    return W1x,W1y
def cond3Stop_p1w_p2w(fn,ngames,alpha=0.05,delta=0.05,newData=False):
    # returns x,y data of p1wins vs p2wins using condition 1 stopping for a max number of games
    read = False
    filename = f"data/{ngames}_{alpha}_{delta}_{fn.__name__}_C3.csv"
    W1x = []

    W1y = []

    import os
    if not newData and os.path.exists(filename):  # then read the file so I dont have to generate again
        print(f"Loading Data for C3 {filename}")
        import csv
        with open(filename, 'r') as csvfile:
            spamreader = csv.reader(csvfile, delimiter=',')
            for row in spamreader:
                if row == []: continue
                x, y = row
                W1x.append(int(x))
                W1y.append(int(y))
        read == True  # don't save it, cause I didnt create anything.
    else:
        print(f"Creating new Data for C3 {filename}")
        for p2w in range(0, ngames):
            for p1w in range(0, ngames):
                n = p1w + p2w

                ###Cond1
                L, U, mean = fn(p1w, n, alpha)
                shouldI1, type = shouldIStop(1, L, U, mean, n, delta=delta)
                L, U, mean = fn(n-p1w, n, alpha)
                shouldI1P2, type = shouldIStop(1, L, U, mean, n, delta=delta)
                if shouldI1P2!=shouldI1:
                    print("Method is assymetric Cond 1")
                if shouldI1: #dont draw over condition1 stopping.
                    #print(f"({p2w},{p1w})")
                    continue #NB This is for C3 Data. If C1 would stop then C3 doesnt activate.
                L, U, mean = fn(p1w, n, alpha/2.0)
                shouldI, type = shouldIStop(3, L, U, mean, n, delta=delta)
                L, U, mean = fn(n-p1w, n, alpha / 2.0)
                shouldIP2, type = shouldIStop(3, L, U, mean, n, delta=delta)
                if shouldIP2!=shouldI:
                    print("Method is assymetric Cond 3")

                if shouldI:
                    W1x.append(p1w)
                    W1y.append(n - p1w)

    if not read: #then write the file so I dont have to generate again
        import csv
        with open(filename, 'w', newline='') as csvfile:
            spamwriter=csv.writer(csvfile,delimiter=',')
            xyval=zip(W1x,W1y)
            for xy in xyval :
                x,y=[*zip(xy)]
                spamwriter.writerow(list(xy))
        print(f"Data written {filename}")

    return W1x,W1y

#############################################################################

#############################################################################
def plot2OnOneFigure(line1Name,line2Name,x1,y1,x2,y2,xlabel,ylabel,name,markersize=.7,filename="",logy=False,show=False,linestyle='',linewidth=1):
    if filename=="":
        filename=f"plots/{name}"
    ###############
    fig, ax = plt.subplots(1, 1, figsize=(19.20, 10.8))
    gcolor = '#b7b7bc'

    ax.grid(color=gcolor, linestyle='-', linewidth=linewidth)
    plt.grid(b=True, which='minor', color=gcolor, linestyle='-', alpha=0.5)
    plt.minorticks_on()

    BPlot, = ax.plot(x1, y1, 'bo', label=fr"{line1Name}", markersize=markersize,linestyle=linestyle,linewidth=linewidth)
    bDraws, = ax.plot(x2, y2, 'ro', label=fr"{line2Name}", markersize=markersize,linestyle=linestyle,linewidth=linewidth)

    ############################################
    # linPlot, = ax.plot(range(0, ngames), range(0, ngames),
    #                   'k-', label='Linear', markersize=markersize)
    ymax = max(list(y1) + list(y2))
    if logy:
        from math import log10, floor
        ax.set_yscale('log', linthreshy=0.001,nonposx='clip')
        plt.yscale('log', linthreshy=0.001)
        ym=round(ymax, -int(floor(log10(abs(ymax)))))
        ax.set_ylim(ymin=1,ymax=ym)
        ax.set_ylim(ymin=1,ymax=ngames)


    else:
        ax.set_ylim(ymin=0, ymax=ngames)
    xmax = max(list(x1) + list(x2))

    ax.set_xlim(xmin=0, xmax=xmax)
        #ax.set_yscale()
        #plt.grid(True)
        #plt.gca().yaxis.grid(True, which='minor', linestyle='-')  # minor grid on too
    ax.grid(True,)
    ax.grid(color=gcolor, which="both",linestyle='-', linewidth=1)
    ax.xaxis.grid(b=True, which='major', color='black', linestyle='-', alpha=0.5)
    ax.xaxis.grid(b=True, which='minor', color=gcolor, linestyle='--', alpha=0.5)
    # ax.tick_params(axis='y', which='minor')
    ax.yaxis.grid(b=True, which='major', color='black', linestyle='-', alpha=0.5)
    ax.yaxis.grid(b=True, which='minor', color=gcolor, linestyle='--', alpha=0.5)


    lgnd = plt.legend(handles=[BPlot, bDraws])
    lgnd.legendHandles[0]._legmarker.set_markersize(markersize * 5)
    lgnd.legendHandles[1]._legmarker.set_markersize(markersize * 5)

    plt.xlabel(fr"{xlabel}")
    plt.ylabel(fr"{ylabel}")
    plt.title(fr"{name}")
    plt.grid(b=True, which='minor', color=gcolor, linestyle='-', alpha=0.5)
    plt.savefig(f"{filename}.eps", format='eps')
    plt.savefig(f"{filename}.png", format='png')
    plt.savefig(f"{filename}.pdf", format='pdf')
    if show:
        plt.show()
    plt.close()

def plot_Aw_vs_Bw_coverage(ngames,alpha=0.05,delta=0.05):
    #creates plot for bayes and wilson.
    ##################################
    prefix="Wins vs Wins"
    W2x, W2y = cond3Stop_p1w_p2w(bayesianU_int, ngames,alpha=alpha,delta=delta,newData=False)
    W1x, W1y = cond1Stop_p1w_p2w(bayesianU_int, ngames,alpha=alpha,delta=delta,)
    name = prefix+"Bayes-U Stopping"
    name=f"{ngames}_{alpha}_{delta}"+name
    xlabel = r"P1Wins"
    ylabel = r"P2Wins"
    line1Name = "Condition 1"
    line2Name = "Condition 2"
    plot2OnOneFigure(line1Name, line2Name, W1x, W1y, W2x, W2y, xlabel, ylabel, name=name,linestyle='',show=False)
    ##################################
    W1x, W1y = cond1Stop_p1w_p2w(bayesianU_int, ngames,alpha=alpha,delta=delta,)
    W2x, W2y = oldC2cond3Stop_p1w_p2w(bayesianU_int, ngames,alpha=alpha,delta=delta,)
    name = prefix+"Bayes-U Stopping C2 Over C1 "
    name = f"{ngames}_{alpha}_{delta}" + name


    plot2OnOneFigure(line1Name, line2Name, W1x, W1y, W2x, W2y, xlabel, ylabel, name=name,linestyle='',show=False)
    ####################################


    ######################################
    Bx = []
    By = []
    W2x, W2y = cond3Stop_p1w_p2w(wils_int, ngames,alpha=alpha,delta=delta,)
    W1x, W1y = cond1Stop_p1w_p2w(wils_int, ngames,alpha=alpha,delta=delta,)
    name = prefix+"Wilson Stopping"
    name = f"{ngames}_{alpha}_{delta}" + name

    plot2OnOneFigure(line1Name,line2Name,W1x, W1y,W2x, W2y,xlabel,ylabel, name=name,linestyle='',show=False)
    #################################### P on X axis ngames on y axis
    ##################################
    W2x, W2y = oldC2cond3Stop_p1w_p2w(wils_int, ngames,alpha=alpha,delta=delta,)
    W1x, W1y = cond1Stop_p1w_p2w(wils_int, ngames,alpha=alpha,delta=delta,)
    name = prefix+"Wilson Stopping C2 Over C1 "
    name = f"{ngames}_{alpha}_{delta}" + name

    plot2OnOneFigure(line1Name, line2Name, W1x, W1y, W2x, W2y, xlabel, ylabel, name=name,linestyle='',show=False)


def plot_PAB_vs_N_coverage(ngames,alpha=0.05,delta=0.05):
    #creates plot for bayes and wilson.
    ##################################
    nsigFigs = 3
    prefix="PvN"
    W2x, W2y = cond3Stop_p1w_p2w(bayesianU_int, ngames,newData=False,alpha=alpha,delta=delta,)
    W1x, W1y = cond1Stop_p1w_p2w(bayesianU_int, ngames,newData=False,alpha=alpha,delta=delta,)
    name = prefix+"bayesStopping"
    name = f"{ngames}_{alpha}_{delta}" + name

    xlabel = "Pab"
    ylabel = "N"
    line1Name = "Condition 1"
    line2Name = "Condition 2"
    combined1=zip(W1x,W1y)
    combined2=zip(W2x,W2y)

    W1x=[]
    W1y=[]
    W2x=[]
    W2y=[]

    x1min={}
    for x1,y1 in combined1:
        n1 = x1 + y1
        x1val=round(x1/n1,nsigFigs)

        y1val = n1
        if x1val in x1min.keys():
            if x1min[x1val]>y1val: #we have a lower value for this value of x
                x1min[x1val]=y1val
        else:
            x1min[x1val] = y1val
    V=list(x1min.items())
    if len(V)>0:
        W1x, W1y=zip(*V)

    x2min={}
    for x2, y2 in combined2:
        n2 = x2 + y2
        x2val = round(x2 / n2, nsigFigs-1)
        y2val = n2
        if x2val in x2min.keys():
            if x2min[x2val] > y2val:  # we have a lower value for this value of x
                x2min[x2val] = y2val
        else:
            x2min[x2val] = y2val
    v=x2min.items()
    if len(v)>0:
        V = list(v)
        W2x, W2y = zip(*V)
    W1S=[]
    if len(W1x) > 0:
        W1S=sorted(zip(W1x, W1y))

    if len(W2x) > 0:
        W2x, W2y=zip(*sorted(zip(W2x, W2y)))
    #now for some magic To remove shading. Note that to get 99% with integers you need to play 100 games.
    # So if left alone it looks like Ngames required increases, but it is just an artifact of the data. Which I remove here.
    # If x<0.5 and the next value is less, then set this value to the next one.
    lastY=1000
    if len(W1x) > 0:
        W1x, W1y=zip(*W1S)
        W1x, W1y=list(W1x), list(W1y)
    L=len(W1x)
    for i in range(1,L):
        if W1x[L-i]<0.5 and W1y[L-i]<W1y[L-i-1]:
          W1y[L-i-1]=W1y[L-i]

        if W1x[i] > 0.5 and W1y[i] > W1y[i-1]:
          W1y[i]=W1y[i - 1]

    #W1x, W1y=W1x[0], W1y[0]
    plot2OnOneFigure(line1Name, line2Name, W1x, W1y, W2x, W2y, xlabel, ylabel,linestyle='', name=name,logy=True,show=False)
    ##################################
    W1x, W1y = cond1Stop_p1w_p2w(bayesianU_int, ngames,alpha=alpha,delta=delta,)
    W2x, W2y = oldC2cond3Stop_p1w_p2w(bayesianU_int, ngames,alpha=alpha,delta=delta,)
    name = prefix+"bayesStopping C2 Over C1"
    name = f"{ngames}_{alpha}_{delta}" + name

    combined1 = zip(W1x, W1y)
    combined2 = zip(W2x, W2y)

    W1x = []
    W1y = []
    W2x = []
    W2y = []

    x1min = {}
    for x1, y1 in combined1:
        n1 = x1 + y1
        x1val = round(x1 / n1, nsigFigs)

        y1val = n1
        if x1val in x1min.keys():
            if x1min[x1val] > y1val:  # we have a lower value for this value of x
                x1min[x1val] = y1val
        else:
            x1min[x1val] = y1val
    V = list(x1min.items())
    if len(V)>0:
        W1x, W1y = zip(*V)

    x2min = {}
    for x2, y2 in combined2:
        n2 = x2 + y2
        x2val = round(x2 / n2, nsigFigs - 1)
        y2val = n2
        if x2val in x2min.keys():
            if x2min[x2val] > y2val:  # we have a lower value for this value of x
                x2min[x2val] = y2val
        else:
            x2min[x2val] = y2val
    v = x2min.items()
    if len(v) > 0:
        V = list(v)
        W2x, W2y = zip(*V)

    W1S = sorted(zip(W1x, W1y))
    if len(W2x)>0:
        W2x, W2y = zip(*sorted(zip(W2x, W2y)))
    # now for some magic To remove shading. Note that to get 99% with integers you need to play 100 games.
    # So if left alone it looks like Ngames required increases, but it is just an artifact of the data. Which I remove here.
    # If x<0.5 and the next value is less, then set this value to the next one.
    lastY = 1000
    W1x, W1y = zip(*W1S)
    W1x, W1y = list(W1x), list(W1y)
    L = len(W1x)
    for i in range(1, L):
        if W1x[L - i] < 0.5 and W1y[L - i] < W1y[L - i - 1]:
            W1y[L - i - 1] = W1y[L - i]

        if W1x[i] > 0.5 and W1y[i] > W1y[i - 1]:
            W1y[i] = W1y[i - 1]
    plot2OnOneFigure(line1Name, line2Name, W1x, W1y, W2x, W2y, xlabel, ylabel,linestyle='',logy=True, name=name,show=False)
    ####################################

    #################################### P on X axis ngames on y axis
    ##################################
    W2x, W2y = cond3Stop_p1w_p2w(wils_int, ngames, alpha=alpha, delta=delta,)
    W1x, W1y = cond1Stop_p1w_p2w(wils_int, ngames, alpha=alpha, delta=delta,)
    name = prefix + "Wilson Stopping"
    name = f"{ngames}_{alpha}_{delta}" + name

    combined1 = zip(W1x, W1y)
    combined2 = zip(W2x, W2y)

    W1x = []
    W1y = []
    W2x = []
    W2y = []

    x1min = {}
    for x1, y1 in combined1:
        n1 = x1 + y1
        x1val = round(x1 / n1, nsigFigs)

        y1val = n1
        if x1val in x1min.keys():
            if x1min[x1val] > y1val:  # we have a lower value for this value of x
                x1min[x1val] = y1val
        else:
            x1min[x1val] = y1val
    V = list(x1min.items())
    if len(V)>0:
        W1x, W1y = zip(*V)

    x2min = {}
    for x2, y2 in combined2:
        n2 = x2 + y2
        x2val = round(x2 / n2, nsigFigs - 1)
        y2val = n2
        if x2val in x2min.keys():
            if x2min[x2val] > y2val:  # we have a lower value for this value of x
                x2min[x2val] = y2val
        else:
            x2min[x2val] = y2val
    v=x2min.items()
    if len(v)>0:
        V = list(v)
        W2x, W2y = zip(*V)
    W1S=[]
    if len(W1x) > 0:
        W1S = sorted(zip(W1x, W1y))
    if len(W2x)>0:
        W2x, W2y = zip(*sorted(zip(W2x, W2y)))
    # now for some magic To remove shading. Note that to get 99% with integers you need to play 100 games.
    # So if left alone it looks like Ngames required increases, but it is just an artifact of the data. Which I remove here.
    # If x<0.5 and the next value is less, then set this value to the next one.
    lastY = 1000
    if len(W1x) > 0:
        W1x, W1y = zip(*W1S)
        W1x, W1y = list(W1x), list(W1y)
    L = len(W1x)
    for i in range(1, L):
        if W1x[L - i] < 0.5 and W1y[L - i] < W1y[L - i - 1]:
            W1y[L - i - 1] = W1y[L - i]

        if W1x[i] > 0.5 and W1y[i] > W1y[i - 1]:
            W1y[i] = W1y[i - 1]
    plot2OnOneFigure(line1Name, line2Name, W1x, W1y, W2x, W2y, xlabel, ylabel,linestyle='', name=name,logy=True,show=False)


    Bx = []
    By = []

    W2x, W2y = oldC2cond3Stop_p1w_p2w(wils_int, ngames, alpha=alpha, delta=delta, )
    W1x, W1y = cond1Stop_p1w_p2w(wils_int, ngames, alpha=alpha, delta=delta, )
    name = prefix + "WilsonStopping_C2 Over C1"
    name = f"{ngames}_{alpha}_{delta}" + name

    combined1 = zip(W1x, W1y)
    combined2 = zip(W2x, W2y)

    W1x = []
    W1y = []
    W2x = []
    W2y = []

    x1min = {}
    for x1, y1 in combined1:
        n1 = x1 + y1
        x1val = round(x1 / n1, nsigFigs)

        y1val = n1
        if x1val in x1min.keys():
            if x1min[x1val] > y1val:  # we have a lower value for this value of x
                x1min[x1val] = y1val
        else:
            x1min[x1val] = y1val
    V = list(x1min.items())
    W1x, W1y = zip(*V)

    x2min = {}
    for x2, y2 in combined2:
        n2 = x2 + y2
        x2val = round(x2 / n2, nsigFigs - 1)
        y2val = n2
        if x2val in x2min.keys():
            if x2min[x2val] > y2val:  # we have a lower value for this value of x
                x2min[x2val] = y2val
        else:
            x2min[x2val] = y2val
    V = list(x2min.items())
    if len(V) > 0:
        W2x, W2y = zip(*V)

    W1S = sorted(zip(W1x, W1y))
    if len(W2x) > 0:
        W2x, W2y = zip(*sorted(zip(W2x, W2y)))

    # This used to say this W2x, W2y = zip(*sorted(zip(W1x, W2y)))
    # now for some magic To remove shading. Note that to get 99% with integers you need to play 100 games.
    # So if left alone it looks like Ngames required increases, but it is just an artifact of the data. Which I remove here.
    # If x<0.5 and the next value is less, then set this value to the next one.
    lastY = 1000
    W1x, W1y = zip(*W1S)
    W1x, W1y = list(W1x), list(W1y)
    L = len(W1x)
    for i in range(1, L):
        if W1x[L - i] < 0.5 and W1y[L - i] < W1y[L - i - 1]:
            W1y[L - i - 1] = W1y[L - i]

        if W1x[i] > 0.5 and W1y[i] > W1y[i - 1]:
            W1y[i] = W1y[i - 1]
    plot2OnOneFigure(line1Name, line2Name, W1x, W1y, W2x, W2y, xlabel, ylabel, linestyle='', logy=True, name=name,
                     show=False)


def plot_PAB_vs_N_coverageShaded(ngames,alpha=0.05,delta=0.05):
    #creates plot for bayes and wilson.
    ##################################
    prefix="PvNShaded"
    W2x, W2y = cond3Stop_p1w_p2w(bayesianU_int, ngames,newData=False,alpha=alpha,delta=delta,)
    W1x, W1y = cond1Stop_p1w_p2w(bayesianU_int, ngames,newData=False,alpha=alpha,delta=delta,)
    name = prefix+"C1_C3bayesStopping_PvN_shaded"
    name = f"{ngames}_{alpha}_{delta}" + name
    xlabel = "Pab"
    ylabel = "N"
    line1Name = "Condition 1"
    line2Name = "Condition 2"
    combined1=zip(W1x,W1y)
    combined2=zip(W2x,W2y)

    W1x=[]
    W1y=[]
    W2x=[]
    W2y=[]

    nsigFigs=8
    for x1,y1 in combined1:
        n1 = x1 + y1
        x1val=round(x1/n1,nsigFigs)
        W1x.append(x1val)
        W1y.append(n1)
    for x1,y1 in combined2:
        n1 = x1 + y1
        x1val=round(x1/n1,nsigFigs)
        W2x.append(x1val)
        W2y.append(n1)

    #now for some magic To remove shading. Note that to get 99% with integers you need to play 100 games.
    # So if left alone it looks like Ngames required increases, but it is just an artifact of the data. Which I remove here.
    # If x<0.5 and the next value is less, then set this value to the next one.


    #W1x, W1y=W1x[0], W1y[0]
    plot2OnOneFigure(line1Name, line2Name, W1x, W1y, W2x, W2y, xlabel, ylabel, name=name,logy=True,show=False,linestyle='')
    ##################################
    W1x, W1y = cond1Stop_p1w_p2w(bayesianU_int, ngames,newData=False,alpha=alpha,delta=delta,)
    W2x, W2y = oldC2cond3Stop_p1w_p2w(bayesianU_int, ngames,newData=False,alpha=alpha,delta=delta,)
    name = prefix+"bayesStopping C2 over C1"
    name = f"{ngames}_{alpha}_{delta}" + name

    combined1 = zip(W1x, W1y)
    combined2 = zip(W2x, W2y)

    W1x = []
    W1y = []
    W2x = []
    W2y = []

    x1min = {}
    for x1,y1 in combined1:
        n1 = x1 + y1
        x1val=round(x1/n1,nsigFigs)
        W1x.append(x1val)
        W1y.append(n1)
    for x1,y1 in combined2:
        n1 = x1 + y1
        x1val=round(x1/n1,nsigFigs)
        W2x.append(x1val)
        W2y.append(n1)

    W1S=sorted(zip(W1x, W1y))
    W1x, W1y=zip(*W1S)

    if len(W2x)>0:
        W2x, W2y=zip(*sorted(zip(W2x, W2y)))
    plot2OnOneFigure(line1Name, line2Name, W1x, W1y, W2x, W2y, xlabel, ylabel, name=name,linestyle='',logy=True)
    ####################################

    ##################################
    W2x, W2y = cond3Stop_p1w_p2w(wils_int, ngames, alpha=alpha, delta=delta, )
    W1x, W1y = cond1Stop_p1w_p2w(wils_int, ngames, alpha=alpha, delta=delta, )
    name = prefix + "WilsStopping"
    name = f"{ngames}_{alpha}_{delta}" + name

    combined1 = zip(W1x, W1y)
    combined2 = zip(W2x, W2y)

    W1x = []
    W1y = []
    W2x = []
    W2y = []

    x1min = {}
    for x1, y1 in combined1:
        n1 = x1 + y1
        x1val = round(x1 / n1, nsigFigs)
        W1x.append(x1val)
        W1y.append(n1)
    for x1, y1 in combined2:
        n1 = x1 + y1
        x1val = round(x1 / n1, nsigFigs)
        W2x.append(x1val)
        W2y.append(n1)

    W1S = sorted(zip(W1x, W1y))
    W1x, W1y = zip(*W1S)
    if len(W2x) > 0:
        W2x, W2y = zip(*sorted(zip(W2x, W2y)))
    plot2OnOneFigure(line1Name, line2Name, W1x, W1y, W2x, W2y, xlabel, ylabel, name=name, linestyle='', logy=True)

    ######################################
    Bx = []
    By = []

    W2x, W2y = oldC2cond3Stop_p1w_p2w(wils_int, ngames,alpha=alpha,delta=delta,)
    W1x, W1y = cond1Stop_p1w_p2w(wils_int, ngames,alpha=alpha,delta=delta,)
    name = prefix+"WilsonStopping C2 Over C1"
    name = f"{ngames}_{alpha}_{delta}" + name

    combined1 = zip(W1x, W1y)
    combined2 = zip(W2x, W2y)

    W1x = []
    W1y = []
    W2x = []
    W2y = []

    x1min = {}
    for x1,y1 in combined1:
        n1 = x1 + y1
        x1val=round(x1/n1,nsigFigs)
        W1x.append(x1val)
        W1y.append(n1)
    for x1,y1 in combined2:
        n1 = x1 + y1
        x1val=round(x1/n1,nsigFigs)
        W2x.append(x1val)
        W2y.append(n1)

    W1S=sorted(zip(W1x, W1y))
    W1x, W1y=zip(*W1S)

    if len(W2x)>0:
        W2x, W2y=zip(*sorted(zip(W2x, W2y)))
    plot2OnOneFigure(line1Name, line2Name, W1x, W1y, W2x, W2y, xlabel, ylabel, name=name,linestyle='',logy=True)
    #################################### P on X axis ngames on y axis


#####################################################################
import os
if __name__ == '__main__':

    ngames=1000
    try:
        os.mkdir("data")
    except FileExistsError:
        pass
    try:
        os.mkdir("plots")
    except FileExistsError:
        pass
    #creates the dataset for when all conditions stop for ngames.

    al=[0.01,0.1,0.05,]
    de=[0.05,0.1,0.01,0.02]

    ng=[300,500,1000,2000,3000]

    for ngames in ng:
        for a in al:
            for d in de:
                plot_Aw_vs_Bw_coverage(ngames, alpha=a, delta=d)
                plot_PAB_vs_N_coverage(ngames, alpha=a, delta=d)
                plot_PAB_vs_N_coverageShaded(ngames, alpha=a, delta=d)
                #######Writes data for plotting later if wanted.

                #W3x, W3y = cond3Stop_p1w_p2w(wils_int, ngames, alpha=a, delta=d, )
                #W2x, W2y = oldC2cond3Stop_p1w_p2w(wils_int, ngames,alpha=a,delta=d,)
                #W1x, W1y = cond1Stop_p1w_p2w(wils_int, ngames,alpha=a,delta=d,)
                #W3x, W3y = cond3Stop_p1w_p2w(bayesianU_int, ngames, alpha=a, delta=d, )
                #W2x, W2y = oldC2cond3Stop_p1w_p2w(bayesianU_int, ngames, alpha=a, delta=d, )
                #W1x, W1y = cond1Stop_p1w_p2w(bayesianU_int, ngames, alpha=a, delta=d, )



    #cameronsPlot(ngames)
