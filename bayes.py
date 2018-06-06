from lib import *
import scipy.stats as stats
import math

def bayesian_U(X,n,alpha=0.05):
        """
        NB alpha=1-confidence
        X successes on n trials
        http://en.wikipedia.org/wiki/Binomial_proportion_confidence_interval
        """
        if X == 0:
            p1L = 0.0
            p1H = 1 - math.pow(alpha,1.0/(n+1))

        elif X == n:
            p1L = math.pow(alpha,1.0/(n+1))
            p1H = 1
        else:
            p1L = stats.beta.ppf(alpha / 2.0, X+1, n - X + 1)
            p1H = stats.beta.ppf(1 - alpha / 2.0, X + 1, n - X+1)
            if p1L>p1H:
                pass

        #if (n-1)%5==0:
        #    print(f"{p1L},{p1H}")
        #    plotBeta(X,n,alpha)

        return p1L, p1H


def plotBeta(X,n,alpha=0.05):
    ##############PLOTTING ONE
    fig, ax = plt.subplots(1, 1)
    a1 = X + 1
    b1 = n - X + 1

    mean, var, skew, kurt = stats.beta.stats(X, b1, moments='mvsk')
    x1 = np.linspace(0,1, 100)
    ax.plot(x1, stats.beta.pdf(x1, a1, b1),
            'r-', lw=5, alpha=0.6, label='p1betaL pdf')
    p2w=n-X
    a = p2w + 1
    b = n - p2w + 1
    ax.plot(x1, stats.beta.pdf(x1, a, b),
            'g-', lw=5, alpha=0.6, label='p2betaU pdf')
    X2 = n-X
    a = X2 + 1
    b = n - X2 + 1
    x2 = np.linspace(0,1, 100)

    #ax.plot(x2, stats.beta.pdf(x2, a, b),'g-', lw=5, alpha=0.6, label='p2beta pdf')
    plt.show()
    pass
import matplotlib.pyplot as plt
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
        p1L,p1U=bayesian_U(p1.nWins,n)

        deltaP1=p1U-p1L
        #print(str(deltaP1))

        if (deltaP1)<bayes_U.stoppingPerc: #it could converge to a solution lower than 50%
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
