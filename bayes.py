from lib import *
import scipy.stats as stats
import math
from scipy.misc import comb

#https://github.com/fonnesbeck/scipy2014_tutorial/blob/master/1_Introduction-to-Bayes.ipynb
P_binom = lambda y, n, p: comb(n, y) * p**y * (1-p)**(n-y) #This function returns the probability of observing $y$ events from $n$ trials, where events occur independently with probability $p$.

def B_pdf(theta,a,b):
    f=math.factorial(a+b-1)/(math.factorial(a-1)*(math.factorial(b-1)))*math.pow(theta,a-1)*math.pow(1-theta,b-1)
    return f

#print(P_binom(3,10,0.5))
def getBetaParams(wins,n):
    priora=10.0
    priorb=10.0
    a = wins + priora
    b = n - wins + priorb
    return a,b
def bayesian_U(X,n):
        """
        NB alpha=1-confidence
        X successes on n trials
        Sample Size for Estimating a Binomial Proportion comparison of different methods Table 1 Gonçalves 2012

        Eq 4 from The cost of using exact confidence intervals for a binomial proportion... Thulin 2013
        http://statweb.stanford.edu/~serban/116/bayes.pdf
        https://alexanderetz.com/2015/07/25/understanding-bayes-updating-priors-via-the-likelihood/
        http://patricklam.org/teaching/bayesianhour_print.pdf p13 and 14
        http://patricklam.org/teaching/conjugacy_print.pdf pp0-13
        https://stats.stackexchange.com/questions/181934/sequential-update-of-bayesian
        NB that a batch update of the posterior is equivalent to a sequential update.
        In reality with this method I am just restarting from the uninformed flat prior each time
        I call this. Posterior is Beta(α+X,β+n−X)
        NB Highest density point (MAX) is (α+X-1)/(α+β+n-2)
        """
        priora=1 #α
        priorb = 1 #β
        global alpha
        a, b = getBetaParams(X, n)

        if X == 0:
            p1L = 0.0
            p1H = 1 - math.pow(alpha,1.0/(n+1))

        elif X == n:
            p1L = math.pow(alpha,1.0/(n+1))
            p1H = 1
        #else:
        #p1LManual=B_pdf(alpha / 2.0, a, b)

        p1L = stats.beta.ppf(alpha / 2.0, a, b) #Percent point function (inverse of cdf — percentiles).
        p1H = stats.beta.ppf(1 - alpha / 2.0, a, b)
        if p1L>p1H:
            pass

        if (n-1)%5==0:
            #print(f"X:{X},n:{n},L:{p1L},U:{p1H}")
            #plotBeta(X,n)
            TTTT=5
        mean, var, skew, kurt = stats.beta.stats(a, b, moments='mvsk')
        #mean=a/(a+b)

        return p1L, p1H,mean

import time

def plotBeta(X,n,text=""):
    ##############PLOTTING ONE
    fig, ax = plt.subplots(1, 1)

    a1,b1=getBetaParams(X,n)

    x1 = np.linspace(0,1, 100)
    #ex=stats.beta(stats.beta,a=a1, b=b1)
    p1Plot,=ax.plot(x1, stats.beta.pdf(x1, a1, b1),
            'r-', lw=5, alpha=0.6, label='p1beta pdf')

    p2w=n-X
    a,b=getBetaParams(p2w,n)

    p2Plot,=ax.plot(x1, stats.beta.pdf(x1, a, b),
            'g-', lw=5, alpha=0.6, label='p2beta pdf')
    mean, var, skew, kurt = stats.beta.stats(a1, b1, moments='mvsk')
    meanT=f"p1Mean={float(mean):.2f}."
    varT=f"\nvar={float(var):.2E}"
    winT=f"\np1Win:{X}, n:{n}"
    conf=f"\na:b:{a1}:{b1}"
    ax.text(0.05, 0.75, meanT+varT+winT, style='italic',
            bbox={'facecolor': 'red', 'alpha': 0.5, 'pad': 10})
    ax.text(0.75, 0.75, text+conf, style='italic',
            bbox={'facecolor': 'red', 'alpha': 0.5, 'pad': 10})

    plt.legend(handles=[p1Plot, p2Plot])

    #ax.plot(x2, stats.beta.pdf(x2, a, b),'g-', lw=5, alpha=0.6, label='p2beta pdf')
    plt.show()
    time.sleep(5) #just becase sciview in pycharm goes through the images v quickly.
    pass
import matplotlib.pyplot as plt

class bayes_MulStop():

    stoppingPerc=.1 #stops when the delta between upper and low is less than this
    name = f"bayes_twoteststop"
    desc = f"bayes_U < {stoppingPerc} " \
           f"delta upper and lower is stopping condition. . " \
           f"Will only predict a winner."

    finished = False
    @staticmethod
    def reset():
        bayes_MulStop.finished=False
    @staticmethod
    def start(p1,p2,drawsP,best_actual):
        #https://www.evanmiller.org/how-not-to-sort-by-average-rating.html
        nwins = p1.nWins
        ngames = p1.nWins + p2.nWins
        totGames = p1.nWins + p2.nWins + drawsP.nWins
        finished=False
        c = None
        lowerBound=0
        if bayes_MulStop.finished:
            return finished, lowerBound, c

        n=float(ngames)
        if n == 0:
            return finished, lowerBound, c
        p1L,p1U,mean=bayesian_U(p1.nWins,n)

        lcp2 = 1 - p1U
        ucp2 = 1 - p1L
        p2L,p2U,mean=bayesian_U(n-p1.nWins,n)

        pred1 = LCBTest(p1L, p1U)
        pred2 = DeltaTest(p1L, p1U)

        if pred1:  # then LCB>0.5
            finished = True
            best_predict = 1 if p1.nWins > p2.nWins else 2
            z = "pred1", p1L, p1U
            if (best_actual != best_predict):
                print("Bayes {pred1} error")
                print(f"{p1L}-{p1U},{p1.nWins}, {p2.nWins}, {drawsP.nWins},{mean}")
                plotBeta(p1.nWins, n,f"Pred1. {p1L:.2f}-{p1U:.2f}")
                if p1.nWins>p2.nWins:
                    jjj=4

        if pred2:
            z = "pred2", p1L, p1U
            best_predict = 1 if p1.nWins > p2.nWins else 2  ###TODO: WHAT ABOUT AROUND 0.5 then its a draw.
            finished = True
            if (best_actual != best_predict):
                print("Bayes {pred2} error")
                print(f"{p1L}-{p1U},{p1.nWins}, {p2.nWins}, {drawsP.nWins},{mean}")
                plotBeta(p1.nWins, n,f"Pred2. {p1L:.2f}-{p1U:.2f}")
            if pred1:
                print("Both tests demand stopping in Bayes")


        if finished:
            #print(f"{p1.nWins},{p2.nWins},{lowerBound},{upperBound}")
            bayes_MulStop.finished=True
            c=conf_stopped_struct(bayes_MulStop.name, [p1L, p1U], p1.nWins, p2.nWins, drawsP.nWins, totGames, best_predict, best_actual, best_actual == best_predict, -1.0)
            #plotBeta(p1.nWins, n, 0.05)
            if best_actual!=best_predict:
                #print(c)
                pass



        return finished,lowerBound,c


class bayes_U():

    stoppingPerc=.1 #stops when the delta between upper and low is less than this
    name = f"bayes_U"
    desc = f"bayes_U < {stoppingPerc} " \
           f"delta upper and lower is stopping condition. . " \
           f"Will only predict a winner."

    finished = False
    @staticmethod
    def reset():
        bayes_U.finished=False
    @staticmethod
    def start(p1,p2,drawsP,best_actual):
        #https://www.evanmiller.org/how-not-to-sort-by-average-rating.html
        nwins = p1.nWins
        ngames = p1.nWins + p2.nWins
        totGames = p1.nWins + p2.nWins + drawsP.nWins
        finished=False
        c = None
        lowerBound=0
        if bayes_U.finished:
            return finished, lowerBound, c

        n=float(ngames)
        if n == 0:
            return finished, lowerBound, c
        p1L,p1U,mean=bayesian_U(p1.nWins,n)

        deltaP1=p1U-p1L
        #print(str(deltaP1))

        if (deltaP1)<bayes_U.stoppingPerc: #it could converge to a solution lower than 50%
            #print('p1 found solution')
            #print(deltaP1)
            finished = True
            if p1.nWins>p2.nWins:
                best_predict = 1
                lowerBound = p1L
                upperBound = p1U
            else:
                best_predict = 2
                lowerBound = p1L
                upperBound = p1U

            if mean>0.5:
                best_predict2 = 1
                lowerBound = p1L
                upperBound = p1U
            if mean < 0.5:
                best_predict2 = 2
                lowerBound = p1L
                upperBound = p1U
            if mean == 0.5:
                best_predict2 = 0
                lowerBound = p1L
                upperBound = p1U
            if best_predict2!=best_predict:
                print("BayesU didnt agree with prediction")
                if best_actual==best_predict2:
                    print("mean predict2 got it right")
                    print(f"{p1L}:{p1U},{p1.nWins}, {p2.nWins}, {drawsP.nWins},{mean}")
                    plotBeta(p1.nWins, n,"Didn't agree. Mean got it")

                else:
                    print("most wins predict2 got it right")
                    print(f"{p1L},{p1U},{p1.nWins}, {p2.nWins}, {drawsP.nWins},{mean}")
                    plotBeta(p1.nWins, n,"Didn't agree. Max got it")
        if finished:
            #print(f"{p1.nWins},{p2.nWins},{lowerBound},{upperBound}")
            bayes_U.finished=True
            c=conf_stopped_struct(bayes_U.name, [lowerBound, upperBound], p1.nWins, p2.nWins, drawsP.nWins, totGames, best_predict, best_actual, best_actual == best_predict, -1.0)
            #plotBeta(p1.nWins, n, 0.05)
            if best_actual!=best_predict:
                #print(c)
                pass



        return finished,lowerBound,c

class bayesTheorum(): #this updates iteratively
    #limit=0.65
    delta=0.3
    minGames=1

    name = f"bayesTheorum_delta={delta}"
    desc = f""

    finished = False
    H_A = "p1 is better"
    H_B = "p2 is better"
    E="Event is p1 wins."
    #####Priors
    priorStart=0.1
    P_A = priorStart  # probably the hypothesis is true. These are updated.
    P_B = priorStart
    ##likelihood
    P_E_A = .51  # Likelihood that given event E the hypothesis is true. These were somewhat plucked. Lower means slower update, but less likely to have type1 errors
    P_E_B = .49

    p1wins = 0  # this keeps track of the wins I know about, cause I just need to know if the last game was a win.
    p2wins = 0
    draws = 0

    @staticmethod
    def reset():
        #####Priors
        bayesTheorum.P_A = bayesTheorum.priorStart
        bayesTheorum.P_B = bayesTheorum.priorStart


        bayesTheorum.p1wins = 0  # this keeps track of the wins I know about, cause I just need to know if the last game was a win.
        bayesTheorum.p2wins = 0
        bayesTheorum.draws = 0
        bayesTheorum.finished=False
    @staticmethod
    def start(p1,p2,drawsP,best_actual):
        #https://www.evanmiller.org/how-not-to-sort-by-average-rating.html
        p1Won=True
        finished = False
        c = None
        if bayesTheorum.finished:
            return finished, 0, c

        #Now find posterior prob
        if p1.nWins!=bayesTheorum.p1wins :
            #then p1 won.
            posA = bayesTheorum.P_E_A * bayesTheorum.P_A
            posB = bayesTheorum.P_E_B * bayesTheorum.P_B
            bayesTheorum.p1wins = p1.nWins
        elif p2.nWins!=bayesTheorum.p2wins:
            #do update for p2 winning.
            posA = (1 - bayesTheorum.P_E_A) * (bayesTheorum.P_A)
            posB = (1 - bayesTheorum.P_E_B) * (bayesTheorum.P_B)
            bayesTheorum.p2wins=p2.nWins
        elif drawsP.nWins!=bayesTheorum.draws: #then it was a draw
            posA = 0.50 * (bayesTheorum.P_A)
            posB = 0.50 * (bayesTheorum.P_B)
            bayesTheorum.draws = drawsP.nWins
            pass
        else:
            assert False #Did you even play a game before calling?
        normSum=posA+posB
        posA=posA/normSum
        posB = posB / normSum
        bayesTheorum.P_A=posA   #These are the latest values
        bayesTheorum.P_B=posB

        lowerBound = 0
        totGames = p1.nWins + p2.nWins #+ drawsP.nWins

        if bayesTheorum.minGames>totGames: #in the case of equal players there might be a large number of draws, so at least give it some good data to decide.
            return finished, lowerBound, c
        if abs(bayesTheorum.P_A-bayesTheorum.P_B)>bayesTheorum.delta:
            finished = True
            if bayesTheorum.P_A>0.50:
                best_predict = 1
            elif bayesTheorum.P_B>0.50:
                best_predict = 2
            else:
                best_predict=0
                print("Predicted a draw")


        if finished:
            #print(f"{p1.nWins},{p2.nWins},{lowerBound},{upperBound}")
            totGames = p1.nWins + p2.nWins + drawsP.nWins
            bayesTheorum.finished=True
            c=conf_stopped_struct(bayesTheorum.name, [bayesTheorum.P_A, bayesTheorum.P_B], p1.nWins, p2.nWins, drawsP.nWins, totGames, best_predict, best_actual, best_actual == best_predict, -1.0)
            if best_actual!=best_predict:
                #print(c)
                pass

        return finished,lowerBound,c
