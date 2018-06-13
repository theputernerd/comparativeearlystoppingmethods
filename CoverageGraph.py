
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


def getLimits(p1w,p2w,fn):
    n=p1w+p2w
    p1L, p1U, mean = fn(p1w, n)
    return p1L, p1U, mean
ngames = 500

def createCoveragePlots(fn,test1=True,test2=True,test3=True):
    x = []
    y = []
    doPred1 = test1
    doPred2 = test2
    doPred3 = test3
    for p2W in range(0, ngames):
        for p1W in range(0, ngames):
            if p2W > p1W:
                continue  # do the other side of the graph later.
            if p2W == 101 and p1W == 153:
                sdfsdf = 44
            p1L, p1U, mean = getLimits(p1W, p2W, fn)
            pred1 = predict1(p1L, p1U) if doPred1 else False
            pred2 = predict2(p1L, p1U) if doPred2 else False
            pred3 = predict3(p1L, p1U) if doPred3 else False

            if pred1 or pred2 or pred3:
                countPredictions += 1
                # predictionMade=True
                x.append(p1W)
                y.append(p2W)
                break
            else:
                # didn't predict so save current data and reset count ready for next check.
                # if predictionMade:
                # need to save the count.
                #    x.append(p1W)
                #    y.append(p2W)
                predictionMade = False
                countPredictions = 0
    x1 = []
    y1 = []
    for p1W in range(0, ngames):
        for p2W in range(0, ngames):
            if p1W > p2W:
                continue  # do the other side of the graph later.
            if p2W > 152 and p1W > 100:
                sdfsdf = 44
            p1L, p1U, mean = getLimits(p1W, p2W, fn)
            pred1 = predict1(p1L, p1U) if doPred1 else False
            pred2 = predict2(p1L, p1U) if doPred2 else False
            pred3 = predict3(p1L, p1U) if doPred3 else False

            if pred1 or pred2:
                countPredictions += 1
                # predictionMade=True
                x.append(p1W)
                y.append(p2W)

                break
            else:
                # didn't predict so save current data and reset count ready for next check.
                # if predictionMade:
                # need to save the count.
                #    x.append(p1W)
                #    y.append(p2W)
                predictionMade = False
                countPredictions = 0
    return x,y

import matplotlib.pyplot as plt
if __name__ == '__main__':
    countPredictions=0
    predictionMade=False
    t1=True
    t2 = True
    t3 = False

    x1,y1=createCoveragePlots(wils_int,t1,t2,t3)
    x,y=createCoveragePlots(bayesian_U,t1,t2,t3)
    x=np.array(x)
    y = np.array(y)
    x1 = np.array(x1)
    y1 = np.array(y1)


    markersize=1
    fig, ax = plt.subplots(1, 2, figsize=(19.20, 10.8))
    gcolor = '#b7b7bc'

    ax[0].grid(color=gcolor, linestyle='-', linewidth=1)
    plt.grid(b=True, which='minor', color=gcolor, linestyle='-', alpha=0.5)
    plt.minorticks_on()

    BPlot, = ax[0].plot(x, y,'go', label="bayes", markersize=markersize)
    WPlot, = ax[0].plot(x1, y1,'bo', label="wilson", markersize=markersize)

    # p2Plot, = ax.plot(x1, y1,
    #                  'b-', label='BayesianP2',markersize =1)
    linPlot, = ax[0].plot(range(0,ngames), range(0,ngames),
                       'k-', label='Linear', markersize=markersize)

    plt.legend(handles=[WPlot, BPlot])
    ax[0].set_ylim(ymin=0)
    ax[0].set_xlim(xmin=0)
    #plt.savefig('destination_path.eps', format='eps')
    #plt.show()
    #fig.canvas.draw()
    #fig.canvas.flush_events()

    gcolor = '#b7b7bc'
    ax[1].grid(color=gcolor, linestyle='-', linewidth=1)
    WPlotngames, = ax[1].plot(x1/(y1+x1), y1+x1,'bo', label="WPlotngames", markersize=markersize)
    BPlotngames, = ax[1].plot(x/(y+x), y+x,'go', label="BPlotngames", markersize=markersize)
    plt.grid(b=True, which='minor', color=gcolor, linestyle='-', alpha=0.5)
    plt.savefig('fig2.eps', format='eps')
    plt.show()
    fig.canvas.draw()
    fig.canvas.flush_events()

