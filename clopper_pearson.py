import math
from lib import *

"""
The cost of using exact condence intervals for a binomial proportion Mans Thulin Department of Mathematics, Uppsala University
Theorum 1.
However, this procedure is necessarily conservative: Article Approximate Is Better than "Exact" for Interval Estimation of Binomial Proportions Agresti, Alan ; Coull, Brent A. The American Statistician, 1 May 1998, Vol.52(2), pp.119-126
https://www.eecs.qmul.ac.uk/~norman/papers/probability_puzzles/bayes_theorem.html
http://www.statisticalengineering.com/bayesian.htm
"""

import scipy.stats as stats

import numpy as np
from scipy import stats

import scipy.stats
import math
def clopper_pearson(X,n,alpha=0.05):
    """
    NB alpha=1-confidence
    X successes on n trials
    http://en.wikipedia.org/wiki/Binomial_proportion_confidence_interval

    """
    if X==0:
        lo=0.0
        hi=1-math.sqrt(alpha/2.0)
    elif X==n:
        lo=math.sqrt(alpha/2.0)
        hi=1
    else:
        lo = scipy.stats.beta.ppf(alpha/2, X, n-X+1)
        hi = scipy.stats.beta.ppf(1 - alpha/2, X+1, n-X) or 1
    return lo, hi

class clopper_pearson_mean_conf():
    """Some authors (Agresti & Coull, 1998; Brown et al., 2001) have argued that
when choosing between condence intervals, it is often preferable to use an interval
with a simple closed-form formula rather than one that requires numerical
evaluation, Thullin - the cost of using exact confidence intervals for a binomial proportion.
"""
    stoppingPerc=.1 #stops when the delta between upper and low is less than this
    name = f"CP_mean_conf"
    desc = f"Cp interval found < {stoppingPerc} " \
           f"delta upper and lower is stopping condition. . " \
           f"Will only predict a winner."

    finished = False
    @staticmethod
    def reset():
        clopper_pearson_mean_conf.finished=False
    @staticmethod
    def start(p1,p2,drawsP,best_actual):
        #https://www.evanmiller.org/how-not-to-sort-by-average-rating.html
        nwins = p1.nWins
        ngames = p1.nWins + p2.nWins
        totGames = p1.nWins + p2.nWins + drawsP.nWins
        finished=False
        c = None
        lowerBound=0
        if clopper_pearson_mean_conf.finished:
            return finished, lowerBound, c

        n=float(ngames)
        if n == 0:
            return finished, lowerBound, c
        p1hat = float(p1.nWins) / n
        p1L,p1U=clopper_pearson(p1.nWins,n)
        if n>40: #closed form formula estimate from theorem 1 The cost of using exact condence intervals for a binomial proportion Mans Thulin 2013
            #NB this is not in use as the full formula is used.
            z=1.96
            #---------------------  Calculate z score and see if it meets threshold.           z = 1.96 #clopper_pearson_mean_conf.zt  # 1.44 = 85%, 1.96 = 95%
            p1hat = float(p1.nWins) / n

            p1L_est=p1hat-(1.0/math.sqrt(n))*z*math.sqrt(p1hat*(1-p1hat))+((2.0*(0.5-p1hat)*z*z-(1+p1hat)))/(3*n)
            p1U_est=p1hat+(1.0/math.sqrt(n))*z*math.sqrt(p1hat*(1-p1hat))+((2.0*(0.5-p1hat)*z*z+(1+p1hat)))/(3*n)
            #((p1hat + z * z / (2 * n) - z * math.sqrt((p1hat * (1 - p1hat) + z * z / (4 * n)) / n)) / (1 + z * z / n))
            #p1U=((p1hat + z * z / (2 * n) + z * math.sqrt((p1hat * (1 - p1hat) + z * z / (4 * n)) / n)) / (1 + z * z / n))

        deltaP1=p1U-p1L
        #print(str(deltaP1))

        if (deltaP1)<clopper_pearson_mean_conf.stoppingPerc: #it could converge to a solution lower than 50%
            #print('p1 found solution')
            #print(deltaP1)
            finished = True
            if ((p1L+p1U)/2.0)>0.5:
                best_predict = 1
                lowerBound = p1L
                upperBound = p1U
            if ((p1L + p1U) / 2.0) < 0.5:
                best_predict = 2
                lowerBound = p1L
                upperBound = p1U
            if ((p1L + p1U) / 2.0) == 0.5:
                best_predict = 0
                lowerBound = p1L
                upperBound = p1U


        if finished:
            #print(f"{p1.nWins},{p2.nWins},{lowerBound},{upperBound}")
            clopper_pearson_mean_conf.finished=True
            c=conf_stopped_struct(clopper_pearson_mean_conf.name, [lowerBound, upperBound], p1.nWins, p2.nWins, drawsP.nWins, totGames, best_predict, best_actual, best_actual == best_predict, -1.0)
            if best_actual!=best_predict:
                #print(c)
                pass

        return finished,lowerBound,c

