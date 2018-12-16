
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
maxNgames = 100  # if the detector hasn't made its mind up by this many games it becomes a type 2 error.
ngames = 350
import numpy as np
import random
import matplotlib.pyplot as plt
def coveragePlotData(fn, test1=True, test2=True):
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

            p1L, p1U, mean = getLimits(p1W, p2W, fn,alpha=0.05)
            pred2 = DeltaTest(p1L, p1U) if doPred2 else False
            pred1 = LCBTest(p1L, p1U) if doPred1 else False
            if pred1 == 1 or pred2 == 1:
                # predicted player 1 won.
                predicted = True
                p1_Lx.append(p1W/(p1W+p2W))
                p1_Ly.append(p1W+p2W)
                #p1_Ux.append(p1W)
                #p1_Uy.append(0)

            elif pred1 == 2 or pred2 == 2:
                #print("Predicted2")
                predicted = True
                p1_Lx.append(p1W / (p1W + p2W))
                p1_Ly.append(p1W + p2W)
                #p1_Lx.append(p1W)
                #p1_Ly.append(p2W)
                #xp2w.append(p1W)
                #yp2w.append(p2W)
                #p1_L.append((p1W,p1W))

            elif pred1 == 3 or pred2 == 3:
                predicted = True
                d_Ux.append(p1W / (p1W + p2W))
                d_Uy.append(p1W + p2W)
                #d_Ux.append(p1W)
                #d_Uy.append(p2W)

                #xdraw.append(p1W)
                #ydraw.append(p2W)
                #p1_L.append((p1W,p1W))

    return p1_Lx,p1_Ly,p1_Ux,p1_Uy,d_Ux,d_Uy


def bayesianCoverageOnly(t1,t2,markersize,name="Bayesian Coverage",sampleEvery=2):
    ###############
    fig, ax = plt.subplots(1, 1, figsize=(19.20, 10.8))
    gcolor = '#b7b7bc'

    ax.grid(color=gcolor, linestyle='-', linewidth=1)
    plt.grid(b=True, which='minor', color=gcolor, linestyle='-', alpha=0.5)
    plt.minorticks_on()

    #####################################################
    p1_Lx, p1_Ly, p1_Ux, p1_Uy, d_Ux, d_Uy = coveragePlotData(bayesian_U, t1, t2)
    p1_Lx, p1_Ly = zip(*random.sample(list(zip(p1_Lx, p1_Ly)), int(len(p1_Lx) / sampleEvery)))
    if len(d_Ux) > 0:
        d_Ux, d_Uy = zip(*random.sample(list(zip(d_Ux, d_Uy)), int(len(d_Ux) / sampleEvery)))

    BPlot, = ax.plot(p1_Lx, p1_Ly, 'bo', label="bayesian", markersize=markersize)
    bDraws, = ax.plot(d_Ux, d_Uy, 'ro', label="bayesian_Draws", markersize=markersize)

    ############################################
    #linPlot, = ax.plot(range(0, ngames), range(0, ngames),
    #                   'k-', label='Linear', markersize=markersize)

    #plt.legend(handles=[BPlot, linPlot, bDraws])
    ax.set_ylim(ymin=0)
    ax.set_xlim(xmin=0)
    plt.xlabel("Player 1 wins")
    plt.ylabel("Player 2 wins")
    plt.title(f"{name}")
    plt.grid(b=True, which='minor', color=gcolor, linestyle='-', alpha=0.5)
    plt.savefig(f"{name}.eps", format='eps')
    plt.show()
    #fig.canvas.draw()
    #fig.canvas.flush_events()
def wCoverageOnly(t1,t2,markersize,name="wils_int Coverage",sampleEvery=2):
    ###############
    fig, ax = plt.subplots(1, 1, figsize=(19.20, 10.8))
    gcolor = '#b7b7bc'

    ax.grid(color=gcolor, linestyle='-', linewidth=1)
    plt.grid(b=True, which='minor', color=gcolor, linestyle='-', alpha=0.5)
    plt.minorticks_on()

    #####################################################
    p1_Lx, p1_Ly, p1_Ux, p1_Uy, d_Ux, d_Uy = coveragePlotData(wils_int, t1, t2)
    p1_Lx, p1_Ly = zip(*random.sample(list(zip(p1_Lx, p1_Ly)), int(len(p1_Lx) / sampleEvery)))
    if len(d_Ux) > 0:
        d_Ux, d_Uy = zip(*random.sample(list(zip(d_Ux, d_Uy)), int(len(d_Ux) / sampleEvery)))

    BPlot, = ax.plot(p1_Lx, p1_Ly, 'bo', label="wils_int", markersize=markersize)
    bDraws, = ax.plot(d_Ux, d_Uy, 'ro', label="wils_int_Draws", markersize=markersize)

    ############################################
    #linPlot, = ax.plot(range(0, ngames), range(0, ngames),
    #                   'k-', label='Linear', markersize=markersize)

    #plt.legend(handles=[BPlot, linPlot, bDraws])
    ax.set_ylim(ymin=0)
    ax.set_xlim(xmin=0)
    plt.xlabel("Player 1 wins")
    plt.ylabel("Player 2 wins")
    plt.title(f"{name}")
    plt.grid(b=True, which='minor', color=gcolor, linestyle='-', alpha=0.5)
    plt.savefig(f"{name}.eps", format='eps')
    plt.show()
    #fig.canvas.draw()
    #fig.canvas.flush_events()

def wilsonCoverageOnly(t1,t2,markersize,name="Wilson Coverage",sampleEvery=2):
    ###############
    fig, ax = plt.subplots(1, 1, figsize=(19.20, 10.8))
    gcolor = '#b7b7bc'

    ax.grid(color=gcolor, linestyle='-', linewidth=1)
    plt.grid(b=True, which='minor', color=gcolor, linestyle='-', alpha=0.5)
    plt.minorticks_on()

    p1_Lx, p1_Ly, p1_Ux, p1_Uy, d_Ux, d_Uy = coveragePlotData(wils_int, t1, t2)
    if len(p1_Lx)>0:
        p1_Lx, p1_Ly=zip(*sorted(zip(p1_Lx, p1_Ly)))
    di={}
    for x,y in zip(p1_Lx, p1_Ly):
        if (x not in di.keys()):
            di[x]=y
        else:
            di[x]=min(di[x],y)
    p1_Lx=[]
    p1_Ly=[]
    for x,y in di.items():
        p1_Lx.append(x)
        p1_Ly.append(y)
    #p1_Ly = p1_Ly[1::sampleEvery]
    #d_Ux = d_Ux[1::sampleEvery]
    #d_Uy = d_Uy[1::sampleEvery]
    if len(d_Ux)>0:
        #d_Ux, d_Uy = zip(*random.sample(list(zip(d_Ux, d_Uy)), int(len(d_Ux)/sampleEvery)))
        d_Ux, d_Uy=zip(*sorted(zip(d_Ux, d_Uy)))

    WPlot, = ax.plot(p1_Lx, p1_Ly, 'bo', label="wilson", markersize=1)
    wDraws, = ax.plot(d_Ux, d_Uy, 'ro', label="wilson_Draws", markersize=1)


    ax.set_ylim(ymin=0)
    ax.set_xlim(xmin=0)
    ax.grid(color='gray', alpha=0.5,linestyle='-', linewidth=2)
    plt.xlabel("PAB")
    plt.ylabel("NGames")
    plt.title(f"{name}")
    plt.grid(b=True, which='minor', color=gcolor, linestyle='-', alpha=0.5)
    plt.savefig(f"{name}.eps", format='eps')
    plt.yscale('linear')
    plt.show()
    #fig.canvas.draw()
    #fig.canvas.flush_events()

def coveragePlot(t1,t2,markersize,name,sampleEvery=2):
    ###############
    fig, ax = plt.subplots(1, 1, figsize=(19.20, 10.8))
    gcolor = '#b7b7bc'

    ax.grid(color=gcolor, linestyle='-', linewidth=1)
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

    WPlot, = ax.plot(p1_Lx, p1_Ly, 'bo', label="wilson", markersize=1)
    wDraws, = ax.plot(d_Ux, d_Uy, 'ro', label="wilson_Draws", markersize=1)

    #####################################################
    p1_Lx, p1_Ly, p1_Ux, p1_Uy, d_Ux, d_Uy = coveragePlotData(bayesian_U, t1, t2)
    p1_Lx, p1_Ly = zip(*random.sample(list(zip(p1_Lx, p1_Ly)), int(len(p1_Lx)/sampleEvery)))
    if len(d_Ux)>0:
        d_Ux, d_Uy = zip(*random.sample(list(zip(d_Ux, d_Uy)), int(len(d_Ux)/sampleEvery)))

    BPlot, = ax.plot(p1_Lx, p1_Ly, 'bo', label="bayesian", markersize=markersize)
    bDraws, = ax.plot(d_Ux, d_Uy, 'ro', label="bayesian_Draws", markersize=markersize)

    ############################################
    #linPlot, = ax.plot(range(0, ngames), range(0, ngames),
    #                   'k-', label='Linear', markersize=markersize)

    #plt.legend(handles=[WPlot, BPlot, linPlot, wDraws, bDraws])
    ax.set_ylim(ymin=0)
    ax.set_xlim(xmin=0)
    plt.xlabel("Player 1 wins")
    plt.ylabel("Player 2 wins")
    plt.title(f"{name}")
    plt.grid(b=True, which='minor', color=gcolor, linestyle='-', alpha=0.5)
    plt.savefig(f"{name}.eps", format='eps')
    plt.show()
    #fig.canvas.draw()
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
def delta_WilsOnly(name="Wilson Only ",sampleEvery=1):
    t1 = False
    t2 = True
    markersize = 1
    wCoverageOnly(t1,t2,markersize,name,sampleEvery=sampleEvery)
def LCB_WilsOnly(name="Wilson Only LCB_UCB Test",sampleEvery=1):
    t1 = True
    t2 = False
    markersize = 1
    wilsonCoverageOnly(t1,t2,markersize,name,sampleEvery=sampleEvery)

if __name__ == '__main__':

    assert False #pVn now can be achieved through 2.
    #delta_WilsOnly(sampleEvery=2)
    delta_BayesOnly(sampleEvery=2)
    delta_WilsOnly(sampleEvery=2)
    #both_TestsTestPlots()

    #LCB_WilsOnly(sampleEvery=2)
    assert False
    delta_BayesOnly(sampleEvery=1)
    LCB_BayesOnly(sampleEvery=1)

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
    BPlot2, = ax[0].plot(b_x2w, b_y2w,'r', label="bayes_p2Wins", markersize=markersize)
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
    plt.savefig('fig2.eps', format='eps')
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

