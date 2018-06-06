import numpy
from scipy.stats import beta
from scipy.stats import norm

def binomial_hpdr(X, N, pct, a=1, b=1, n_pbins=1e3): ##https://stackoverflow.com/questions/13059011/is-there-any-python-function-library-for-calculate-binomial-confidence-intervals
    """
    Function computes the posterior mode along with the upper and lower bounds of the
    **Highest Posterior Density Region**.

    Parameters
    ----------
    X: number of successes
    N: sample size
    pct: the size of the confidence interval (between 0 and 1)
    a: the alpha hyper-parameter for the Beta distribution used as a prior (Default=1)
    b: the beta hyper-parameter for the Beta distribution used as a prior (Default=1)
    n_pbins: the number of bins to segment the p_range into (Default=1e3)

    Returns
    -------
    A tuple that contains the mode as well as the lower and upper bounds of the interval
    (mode, lower, upper)

    """
    # fixed random variable object for posterior Beta distribution
    rv = beta(X + a, N - X + b)
    # determine the mode and standard deviation of the posterior
    stdev = rv.stats('v')**0.5
    mode = (X + a - 1.) / (N + a + b - 2.)
    # compute the number of sigma that corresponds to this confidence
    # this is used to set the rough range of possible success probabilities
    n_sigma = numpy.ceil(norm.ppf( (1+pct)/2. ))+1
    # set the min and max values for success probability
    max_p = mode + n_sigma * stdev
    if max_p > 1:
        max_p = 1.
    min_p = mode - n_sigma * stdev
    if min_p > 1:
        min_p = 1.
    # make the range of success probabilities
    p_range = numpy.linspace(min_p, max_p, n_pbins+1)
    # construct the probability mass function over the given range
    if mode > 0.5:
        sf = rv.sf(p_range)
        pmf = sf[:-1] - sf[1:]
    else:
        cdf = rv.cdf(p_range)
        pmf = cdf[1:] - cdf[:-1]
    # find the upper and lower bounds of the interval
    sorted_idxs = numpy.argsort( pmf )[::-1]
    cumsum = numpy.cumsum( numpy.sort(pmf)[::-1] )
    j = numpy.argmin( numpy.abs(cumsum - pct) )
    upper = p_range[ (sorted_idxs[:j+1]).max()+1 ]
    lower = p_range[ (sorted_idxs[:j+1]).min() ]

    return (mode, lower, upper)

import math
from lib import *

class binomial_mean_conf():

    stoppingPerc=.1 #stops when the delta between upper and low is less than this
    name = f"binomial_hpdr"
    desc = f"binomial_hpdr interval found < {stoppingPerc} " \
           f"delta upper and lower is stopping condition. . " \
           f"Will only predict a winner."

    finished = False
    @staticmethod
    def reset():
        binomial_mean_conf.finished=False
    @staticmethod
    def start(p1,p2,drawsP,best_actual):
        #https://www.evanmiller.org/how-not-to-sort-by-average-rating.html
        nwins = p1.nWins
        ngames = p1.nWins + p2.nWins
        totGames = p1.nWins + p2.nWins + drawsP.nWins
        finished=False
        c = None
        lowerBound=0
        if binomial_mean_conf.finished:
            return finished, lowerBound, c

        n=float(ngames)
        if n == 0:
            return finished, lowerBound, c
        p1hat = float(p1.nWins) / n
        #a=stats.binom.stats(n, p1hat, moments='mvsk')
        mode, p1L, p1U=binomial_hpdr(p1.nWins,n,.95)

        deltaP1=p1U-p1L
        #print(str(deltaP1))

        if (deltaP1)<binomial_mean_conf.stoppingPerc: #it could converge to a solution lower than 50%
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
            binomial_mean_conf.finished=True
            c=conf_stopped_struct(binomial_mean_conf.name, [lowerBound, upperBound], p1.nWins, p2.nWins, drawsP.nWins, totGames, best_predict, best_actual, best_actual == best_predict, -1.0)
            if best_actual!=best_predict:
                #print(c)
                pass

        return finished,lowerBound,c

