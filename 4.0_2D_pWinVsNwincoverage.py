
from wilson import wils_int,wilson_z_score_0min,wilson_lcb,wilson_conf_delta,wilson_z_score_100min,wilson_z_score_30min,wilsonNOCC_z_score_30min,wilsonNOCC_z_score_100min
from wald import wal_z_score_30min,wal_z_score_100min, wal_conf_delta
from naiveMethods import perc_after_n_games,n_games
from timeit import default_timer as timer
from clopper_pearson import clopper_pearson_mean_conf
from agresti_coull import ac_z_score_30min,ac_z_score_100min
from lib import *
from lib import *
import time
from binomial import binomial_mean_conf
from bayes import bayesTheorum,bayes_U,bayes_MulStop,bayesian_U
import csv
ngames = 300
linewidth=1
import numpy as np
import random
import matplotlib.pyplot as plt
def coveragePlotData(fn, test1=True, test2=True,alpha=0.05,delta=0.05):
    doPred1 = test1
    doPred2 = test2
    p1_Lx=[] #the lower bound of p1 winning
    p1_Ly=[] #the lower bound of p1 winning
    p1_Ux = []  # the lower bound of p1 winning
    p1_Uy = []  # the lower bound of p1 winning
    d_Ux = []  # the lower bound of p1 winning
    d_Uy = []  # the lower bound of p1 winning
    for p2W in range(0, ngames):
        predicted = False
        for p1W in range(0, ngames):
            p1L, p1U, mean = fn(p1W, p2W+p1W, alpha=alpha)
            p = shouldIStop(1, p1L, p1U, mean, n, delta=0) if doPred1 else (False, 0)
            (pred1, t1)=p[0],p[1]
            p1L2, p1U2, mean = fn(p1W, p2W+p1W, alpha=alpha/2.0)
            p = shouldIStop(3, p1L2, p1U2, mean, n, delta=delta) if doPred2 else (False, 0)
            (pred2, t2)=p[0],p[1]

            if t1 == 1 or t2 == 1:
                # predicted player 1 won.
                predicted = True
                p1_Lx.append(p1W)
                p1_Ly.append(p2W)
                #p1_Ux.append(p1W)
                #p1_Uy.append(0)

            elif t1 == 2 or t2 == 2:
                predicted = True
                p1_Lx.append(p1W)
                p1_Ly.append(p2W)
                #xp2w.append(p1W)
                #yp2w.append(p2W)
                #p1_L.append((p1W,p1W))

            elif t1 == 3 or t2 == 3:
                predicted = True
                d_Ux.append(p1W)
                d_Uy.append(p2W)

                #xdraw.append(p1W)
                #ydraw.append(p2W)
                #p1_L.append((p1W,p1W))

    return p1_Lx,p1_Ly,p1_Ux,p1_Uy,d_Ux,d_Uy


def bayesianCoverageOnly(t1,t2,markersize,name="Bayesian Coverage",filename="",sampleEvery=2,alpha=0.05,delta=0.1):
    ###############
    fig, ax = plt.subplots(1, 1, figsize=(19.20, 10.8))
    gcolor = '#b7b7bc'

    ax.grid(color=gcolor, linestyle='-', linewidth=linewidth)
    plt.grid(b=True, which='minor', color=gcolor, linestyle='-', alpha=0.5)
    plt.minorticks_on()

    #####################################################
    p1_Lx, p1_Ly, p1_Ux, p1_Uy, d_Ux, d_Uy = coveragePlotData(bayesian_U, t1, t2,alpha=alpha,delta=delta)
    p1_Lx, p1_Ly = zip(*random.sample(list(zip(p1_Lx, p1_Ly)), int(len(p1_Lx) / sampleEvery)))
    if len(d_Ux) > 0:
        d_Ux, d_Uy = zip(*random.sample(list(zip(d_Ux, d_Uy)), int(len(d_Ux) / sampleEvery)))

    BPlot, = ax.plot(p1_Lx, p1_Ly, 'bo', label="Condition 1", markersize=markersize)
    bDraws, = ax.plot(d_Ux, d_Uy, 'go', label="Condition 2", markersize=markersize)

    ############################################
    #linPlot, = ax.plot(range(0, ngames), range(0, ngames),
    #                   'k-', label='Linear', markersize=markersize)

    lgnd=plt.legend(handles=[BPlot, bDraws])
    lgnd.legendHandles[0]._legmarker.set_markersize(markersize*5)
    lgnd.legendHandles[1]._legmarker.set_markersize(markersize*5)
    ax.set_ylim(ymin=0,ymax=ngames)
    ax.set_xlim(xmin=0,xmax=ngames)
    plt.xlabel(r"Player A wins, $k$")
    plt.ylabel(r"Player B wins, $n-k$")
    plt.title(f"{name}")
    plt.grid(b=True, which='minor', color=gcolor, linestyle='-', alpha=0.5)
    plt.savefig(f"4.0_2D_pWinVsNwincoverage\{filename}.eps", format='eps')
    plt.savefig(f"4.0_2D_pWinVsNwincoverage\{filename}.png", format='png')
    plt.savefig(f"4.0_2D_pWinVsNwincoverage\{filename}.pdf", format='pdf')
    plt.show()
    #fig.canvas.draw()
    #fig.canvas.flush_events()

def wilsonCoverageOnly(t1,t2,markersize,name="Wilson Coverage",filename="",sampleEvery=2,alpha=0.05,delta=0.1):
    ###############
    fig, ax = plt.subplots(1, 1, figsize=(19.20, 10.8))
    gcolor = '#b7b7bc'

    ax.grid(color=gcolor, linestyle='-', linewidth=linewidth)
    plt.grid(b=True, which='minor', color=gcolor, linestyle='-', alpha=0.5)
    plt.minorticks_on()

    p1_Lx, p1_Ly, p1_Ux, p1_Uy, d_Ux, d_Uy = coveragePlotData(wils_int, t1, t2,alpha=alpha,delta=delta)
    if len(p1_Lx)>0:

        p1_Lx, p1_Ly = zip(*random.sample(list(zip(p1_Lx, p1_Ly)), int(len(p1_Lx)/sampleEvery)))

    #p1_Ly = p1_Ly[1::sampleEvery]
    #d_Ux = d_Ux[1::sampleEvery]
    #d_Uy = d_Uy[1::sampleEvery]
    if len(d_Ux)>0:
        d_Ux, d_Uy = zip(*random.sample(list(zip(d_Ux, d_Uy)), int(len(d_Ux)/sampleEvery)))

    WPlot, = ax.plot(p1_Lx, p1_Ly, 'bo', label="Condition 1", markersize=markersize)
    wDraws, = ax.plot(d_Ux, d_Uy, 'go', label="Condition 2", markersize=markersize)


    ############################################
    #linPlot, = ax.plot(range(0, ngames), range(0, ngames),
    #                   'k-', label='Linear', markersize=markersize)

    lgnd = plt.legend(handles=[WPlot, wDraws])
    lgnd.legendHandles[0]._legmarker.set_markersize(markersize * 5)
    lgnd.legendHandles[1]._legmarker.set_markersize(markersize * 5)
    ax.set_ylim(ymin=0, ymax=ngames)
    ax.set_xlim(xmin=0, xmax=ngames)
    plt.xlabel(r"Player A wins, $k$")
    plt.ylabel(r"Player B wins, $n-k$")
    plt.title(f"{name}")
    plt.grid(b=True, which='minor', color=gcolor, linestyle='-', alpha=0.5)
    plt.savefig(f"4.0_2D_pWinVsNwincoverage\{filename}.eps", format='eps')
    plt.savefig(f"4.0_2D_pWinVsNwincoverage\{filename}.png", format='png')
    plt.savefig(f"4.0_2D_pWinVsNwincoverage\{filename}.pdf", format='pdf')
    plt.show()
    #fig.canvas.draw()
    #fig.canvas.flush_events()

def coveragePlot(t1,t2,markersize,name,sampleEvery=2):
    ###############
    fig, ax = plt.subplots(1, 1, figsize=(19.20, 10.8))
    gcolor = '#b7b7bc'

    ax.grid(color=gcolor, linestyle='-', linewidth=linewidth)
    plt.grid(b=True, which='minor', color=gcolor, linestyle='-', alpha=0.5)
    plt.minorticks_on()

    p1_Lx, p1_Ly, p1_Ux, p1_Uy, d_Ux, d_Uy = coveragePlotData(wils_int, t1, t2)
    if len(p1_Lx)>0:

        p1_Lx, p1_Ly = zip(*random.sample(list(zip(p1_Lx, p1_Ly)), int(len(p1_Lx)/sampleEvery)))

    #p1_Ly = p1_Ly[1::sampleEvery]
    #d_Ux = d_Ux[1::sampleEvery]
    #d_Uy = d_Uy[1::sampleEvery]
    if len(d_Ux)>0:
        d_Ux, d_Uy = zip(*random.sample(list(zip(d_Ux, d_Uy)), int(len(d_Ux)/sampleEvery)))

    WPlot, = ax.plot(p1_Lx, p1_Ly, 'bo', label="wilson", markersize=markersize)
    wDraws, = ax.plot(d_Ux, d_Uy, 'go', label="wilson_Draws", markersize=markersize)

    #####################################################
    p1_Lx, p1_Ly, p1_Ux, p1_Uy, d_Ux, d_Uy = coveragePlotData(bayesian_U, t1, t2)
    p1_Lx, p1_Ly = zip(*random.sample(list(zip(p1_Lx, p1_Ly)), int(len(p1_Lx)/sampleEvery)))
    if len(d_Ux)>0:
        d_Ux, d_Uy = zip(*random.sample(list(zip(d_Ux, d_Uy)), int(len(d_Ux)/sampleEvery)))

    BPlot, = ax.plot(p1_Lx, p1_Ly, 'ro', label="bayesian", markersize=markersize)
    bDraws, = ax.plot(d_Ux, d_Uy, 'mo', label="bayesian_Draws", markersize=markersize)

    ############################################
    #linPlot, = ax.plot(range(0, ngames), range(0, ngames),
    #                   'k-', label='Linear', markersize=markersize)

    plt.legend(handles=[WPlot, BPlot, wDraws, bDraws])
    ax.set_ylim(ymin=0)
    ax.set_xlim(xmin=0)
    plt.xlabel("Player 1 wins")
    plt.ylabel("Player 2 wins")
    plt.title(f"{name}")
    plt.grid(b=True, which='minor', color=gcolor, linestyle='-', alpha=0.5)
    plt.savefig(f"4.0_2D_pWinVsNwincoverage\{name}.eps", format='eps')
    plt.savefig(f"4.0_2D_pWinVsNwincoverage\{name}.png", format='png')
    plt.savefig(f"4.0_2D_pWinVsNwincoverage\{name}.pdf", format='pdf')
    plt.show()
    fig.canvas.draw()
    #fig.canvas.flush_events()


def LCBTestPlots(name="0.5<LCB 0.5>UCB "):
    countPredictions = 0
    predictionMade = False
    t1 = True
    t2 = False
    markersize = 1
    coveragePlot(t1,t2,markersize,name,sampleEvery=1)


def both_TestsTestPlots(name="both__TestCoverage"):
    countPredictions = 0
    predictionMade = False
    t1 = True
    t2 = True
    markersize = 1
    ###############
    coveragePlot(t1,t2,markersize,name)



def Delta_CBTestPlots(name="Distribution Width Coverage"):
    countPredictions = 0
    predictionMade = False
    t1 = False
    t2 = True
    markersize = 1
    coveragePlot(t1,t2,markersize,name,sampleEvery=1)

def delta_BayesOnly(name="Bayesian Only ",sampleEvery=1):
    t1 = False
    t2 = True
    markersize = 1
    bayesianCoverageOnly(t1,t2,markersize,name,sampleEvery=sampleEvery)
def LCB_BayesOnly(name="Bayesian Only LCB_UCB Test",sampleEvery=1):
    t1 = True
    t2 = False
    markersize = 1
    bayesianCoverageOnly(t1,t2,markersize,name,sampleEvery=sampleEvery)
def delta_WilsOnly(name="Condition 2 Wilson Only ",sampleEvery=1):
    t1 = False
    t2 = True
    markersize = 1
    wilsonCoverageOnly(t1,t2,markersize,name,sampleEvery=sampleEvery)
def LCB_WilsOnly(name="Wilson Only LCB_UCB Test",sampleEvery=1):
    t1 = True
    t2 = False
    markersize = 1
    wilsonCoverageOnly(t1,t2,markersize,name,sampleEvery=sampleEvery)

if __name__ == '__main__':
    a=0.05
    a=0.0423
    d=.1
    #d=0.05
    markersize=.75
    name=fr"$\alpha={a}, \Delta={d}$"
    filename=f"a_{a}d_{d}"
    plt.rcParams.update({'font.size': 22})
    bayesianCoverageOnly(True, True, markersize, rf"Bayesian Updating {name}", filename=f"Bayesian{filename}", sampleEvery=1,
                         alpha=a, delta=d)
    wilsonCoverageOnly(True, True, markersize, fr"Wilson {name}", filename=f"Wilson{filename}", sampleEvery=1, alpha=a,
                       delta=d)
    assert False

    try:
        bayesianCoverageOnly(True, True,markersize,rf"Bayesian {name}",filename=f"Bayesian {filename}",sampleEvery=1,alpha=a,delta=d)
    except:
        pass
    try:
        wilsonCoverageOnly(True, True, markersize, fr"Wilson {name}",filename=f"Wilson {filename}", sampleEvery=1,alpha=a,delta=d)
    except:
        pass
    assert False
    delta_WilsOnly(sampleEvery=2)
    LCB_WilsOnly(sampleEvery=2)
    delta_BayesOnly(sampleEvery=2)
    LCB_BayesOnly(sampleEvery=2)

    Delta_CBTestPlots()
    LCBTestPlots()

    both_TestsTestPlots()


    assert False
    ########################################################
    x1w,y1w,x2w,y2w,xdraw,ydraw=createCoveragePlots(wils_int,t1,t2)
    b_x1w, b_y1w, b_x2w, b_y2w, b_xdraw, b_ydraw=createCoveragePlots(bayesian_U,t1,t2)



    fig, ax = plt.subplots(1, 2, figsize=(19.20, 10.8))
    gcolor = '#b7b7bc'

    ax[0].grid(color=gcolor, linestyle='-', linewidth=1)
    plt.grid(b=True, which='minor', color=gcolor, linestyle='-', alpha=0.5)
    plt.minorticks_on()

    BPlot, = ax[0].plot(b_x1w, b_y1w,'bo', label="bayes_p1Wins", markersize=markersize)
    BPlot2, = ax[0].plot(b_x2w, b_y2w,'go', label="bayes_p2Wins", markersize=markersize)
    BPlot3, = ax[0].plot(b_xdraw, b_ydraw,'co', label="bayes_Draws", markersize=markersize)

    WPlot, = ax[0].plot(x1w, y1w, 'mo', label="Wils_p1Wins", markersize=markersize)
    WPlot2, = ax[0].plot(x2w, y2w, 'yo', label="Wils_p2Wins", markersize=markersize)
    WPlot3, = ax[0].plot(xdraw, ydraw, 'ko', label="Wils_Draws", markersize=markersize)

    # p2Plot, = ax.plot(x1, y1,
    #                  'b-', label='BayesianP2',markersize =1)
    #linPlot, = ax[0].plot(range(0,ngames), range(0,ngames),
    #                   'k-', label='Linear', markersize=markersize)

    plt.legend(handles=[WPlot, BPlot,BPlot2,BPlot3])
    ax[0].set_ylim(ymin=0)
    ax[0].set_xlim(xmin=0)
    #plt.savefig('destination_path.eps', format='eps')
    #plt.show()
    #fig.canvas.draw()
    #fig.canvas.flush_events()

    gcolor = '#b7b7bc'
    ax[1].grid(color=gcolor, linestyle='-', linewidth=1)
    #WPlotngames, = ax[1].plot(x1/(y1+x1), y1+x1,'bo', label="WPlotngames", markersize=markersize)
    #BPlotngames, = ax[1].plot(x/(y+x), y+x,'go', label="BPlotngames", markersize=markersize)
    plt.grid(b=True, which='minor', color=gcolor, linestyle='-', alpha=0.5)
    plt.savefig('4.0_2D_pWinVsNwincoverage/fig2.eps', format='eps')
    plt.show()
    fig.canvas.draw()
    fig.canvas.flush_events()

def getCoverageData(fn, test1=True, test2=True):
    assert False #NOT USED
    # xp1w = []
    yp1w = []
    xp2w = []
    yp2w = []
    xdraw = []
    ydraw = []

    doPred1 = test1
    doPred2 = test2
    line1=[]
    line2=[]
    for p2W in range(0, ngames):
        predicted = False
        for p1W in range(0, ngames):
            if p2W > p1W:
                continue  # do the other side of the graph later.

            p1L, p1U, mean = getLimits(p1W, p2W, fn)
            pred1 = LCBTest(p1L, p1U) if doPred1 else False
            pred2 = DeltaTest(p1L, p1U) if doPred2 else False

            if pred1==1 or pred2==1:
                #predicted player 1 won.

                predicted=True

                xp1w.append(p1W)
                yp1w.append(p2W)
                break
            elif pred1==2 or pred2==2:
                xp2w.append(p1W)
                yp2w.append(p2W)
                break
            elif pred1 == 3 or pred2==3:
                xdraw.append(p1W)
                ydraw.append(p2W)
                break

    for p1W in range(0, ngames): #ngames
        for p2W in range(0, ngames):
            if p1W > p2W:
                continue  # do the other side of the graph later.
            if p2W > 152 and p1W > 100:
                sdfsdf = 44
            p1L, p1U, mean = getLimits(p1W, p2W, fn)
            pred1 = LCBTest(p1L, p1U) if doPred1 else False
            pred2 = DeltaTest(p1L, p1U) if doPred2 else False

            if pred1==1 or pred2==1:
                #predicted player 1 won.
                xp1w.append(p1W)
                yp1w.append(p2W)
                break
            elif pred1==2 or pred2==2:
                xp2w.append(p1W)
                yp2w.append(p2W)
                break
            elif pred1 == 3 or pred2==3:
                xdraw.append(p1W)
                ydraw.append(p2W)
                break
            else:
                # didn't predict so save current data and reset count ready for next check.
                # if predictionMade:
                # need to save the count.
                #    x.append(p1W)
                #    y.append(p2W)
                predictionMade = False
                countPredictions = 0

    return xp1w,yp1w,xp2w,yp2w,xdraw,ydraw

