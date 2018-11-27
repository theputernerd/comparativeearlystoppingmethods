from collections import namedtuple

conf_stopped_struct = namedtuple("conf_struct", "name score p1wins p2wins draws ngames bestPlayerPredict actualBestPlayer predictCorrect time")
from scipy import stats
import numpy as np
alpha=0.05
z=1.96 #abs(stats.norm.ppf((1-alpha/2.0))) #1.96 for alpha=0.5


def getLimits(p1w, p2w, fn,alpha):
    n = p1w + p2w

    p1L, p1U, mean = fn(p1w, n,alpha)

    return p1L, p1U, mean


def LCBTest(lc, uc):
    if lc>=0.50 :
        return 1
    elif uc<=0.50:
        return 2
    else:
        return 0

def DeltaTest(lc, uc):
    delta=uc-lc
    diff=0.10 #5% diff between upper and lower confidence.
    mean=(uc+lc)/2.0

    if delta<diff:
        if abs(mean - 0.5) <= 0.025:  # this is classed as a draw
            return 3
        if mean > 0.5:
            return 1
        elif mean<0.5:
            return 2
        else:
            assert False #Wierd result.

    else:
        return 0
    if percDiff <diff : ##percentage difference. NB Returns different results for lower and upper confidence because of %diff not absolute
        return True #remove for lc>uc

        lcp2 = 1 - uc
        ucp2 = 1 - lc
        if lc > ucp2:
            return True
        elif lcp2 > uc:
            return True
        else:
            return False

    else:
        return False


def predict3(lc, uc):
    delta = abs(lc - uc)
    diff = 0.10  # 5% diff between upper and lower confidence.
    percDiff = delta / (abs(uc + lc) / 2.0)
    if delta<diff:
        if (lc + uc) / 2.0 > 0.5:
            return True
        else:
            return False
    else:
        return False



class game(object):  #just chooses a random winner based on the probabilty distribution.
    def __init__(self, p1, p2, drawsP):
        self.winner = None
        self.p1 = p1
        self.p2 = p2
        self.drawsP = drawsP
        assert (p1.pWin + p2.pWin + drawsP.pWin) == 1
        pass

    def playGame(self):

        winner = np.random.choice([self.p1, self.p2, self.drawsP], p=[self.p1.pWin, self.p2.pWin, self.drawsP.pWin])
        return winner

import math
def shouldIStop(method, lc, uc, mean, epsilon=0, delta=0.05): #first number is winner, second number stopping condition.
    #method is needed cause lc and uc should be either 0.025 or 0.05
    if epsilon!=0:
        print("Epsilon not zero")
    ut=0.5+epsilon #upperthreshold
    lt=0.5-epsilon #lower threshold
    #lc=int(lc*1000)/1000.0#round down. #covers for when python has rounding errors. Floats cause issues at the boundaries.
    #uc = np.round(uc, 3)
    #mean=np.round(mean,3)

    if method==1:

        if lc>ut:
            return True,1  #player wins from condition 1
        elif uc<lt:
            return True,2 #player loses from condition 1.1
    elif method==2: #Bot UC and LC within +-delta
        assert False
        if uc<=0.5+delta and lc>=0.5-delta:
            return True,3

    elif method == 3:
        diff=math.fabs(uc-lc)
        if diff<=delta:# and any([uc <= 0.5 + delta, lc >= 0.5 - delta]):
            return True, 3

    return (False,66)

def shouldIStopAND(method, lc, uc, mean, epsilon=0, delta=0.05): #first number is winner, second number stopping condition.
    #method is needed cause lc and uc should be either 0.025 or 0.05
    ut=0.5+epsilon #upperthreshold
    lt=0.5-epsilon #lower threshold
    lc=int(lc*1000)/1000.0#round down. #covers for when python has rounding errors. Floats cause issues at the boundaries.
    uc = np.round(uc, 3)
    mean=np.round(mean,3)

    if method==1:
        #ut = 0.5 # upperthreshold
        #lt = 0.5
        ###########################################################
        #ADDED FOR AND ONLY REMOVE FOR NORMAL TEST
        if uc <= 0.5 + delta and lc >= 0.5 - delta:
            return 3, 2.1
        ###########################################################
            if lc>ut:
                return 1,1.1  #player wins from condition 1
            elif uc<lt:
                return 2,1.2 #player loses from condition 1.1
    elif method==2:
        if uc<=0.5+delta and lc>=0.5-delta:
            return 3,2.1
        #if math.fabs(uc-lc)<delta:#NB that predict some # inside the threshold to account for errors on the edge
        #    if mean>ut:
        #        return 1, 2.1  # player wins from condition 2.1
        #    if mean<lt:
        #        return 2, 2.2  # player loses from condition 2.2
        #    if mean>lt and mean<ut:
        #        return 3, 2.3 #it was a draw
    return 0,0

def wils_int(X, n, alpha, cc=True):  ##cc==True means with continuity correction.
    # z_calced=abs(stats.norm.ppf((1-alpha/2.0))) #two tailed.
    # global z
    # z = abs(stats.norm.ppf((1 - alpha / 2.0)))  # 1.96 for alpha=0.5
    z = abs(stats.norm.ppf((1 - alpha)))

    if n == 0:
        return 0, 1, 0.5
    p_hat = float(X) / n
    q = 1 - p_hat

    if cc:
        try:
            if p_hat == 0:
                p1L = 0
            else:
                p1L = max(0, (2 * n * p_hat + z * z - (
                            z * math.sqrt(z * z - 1 / n + 4 * n * p_hat * (1 - p_hat) + (4 * p_hat - 2)) + 1)) / (
                                      2 * (n + z * z)))
            if p_hat == 1:
                p1U = 1
            else:
                p1U = min(1, (2 * n * p_hat + z * z + (
                            z * math.sqrt(z * z - 1 / n + 4 * n * p_hat * (1 - p_hat) - (4 * p_hat - 2)) + 1)) / (
                                      2 * (n + z * z)))

        except ValueError:
            print(f"X:{X},n:{n},p_hat:{p_hat},z:{z}")
            raise
    else:
        p1L = ((p_hat + z * z / (2 * n) - z * math.sqrt((p_hat * (1 - p_hat) + z * z / (4 * n)) / n)) / (
                    1 + z * z / n))
        p1U = ((p_hat + z * z / (2 * n) + z * math.sqrt((p_hat * (1 - p_hat) + z * z / (4 * n)) / n)) / (
                    1 + z * z / n))
    E = n * p_hat
    mean = (p1U + p1L) / 2.0
    return p1L, p1U, mean
def getBetaParams(wins,n):
    priora=5 # 1#10.0
    priorb=5 #1#10.0
    a = wins + priora
    b = n - wins + priorb
    return a,b
def bayesianU_int(X, n, alpha):
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

        a, b = getBetaParams(X, n)
        """
                priora=10.0
                priorb=10.0
                a = wins + priora
                b = n - wins + priorb
                return a,b
        """
        if X == 0:
            p1L = 0.0
            p1H = 1 - math.pow(alpha, 1.0 / (n + 1))

        elif X == n:
            p1L = math.pow(alpha, 1.0 / (n + 1))
            p1H = 1
        # else:
        # p1LManual=B_pdf(alpha / 2.0, a, b)

        p1L = stats.beta.ppf(alpha, a, b)  # Percent point function (inverse of cdf — percentiles).
        p1H = stats.beta.ppf(1 - alpha, a, b)
        if p1L > p1H:
            pass

        if (n - 1) % 5 == 0:
            # print(f"X:{X},n:{n},L:{p1L},U:{p1H}")
            # plotBeta(X,n)
            TTTT = 5
        mean, var, skew, kurt = stats.beta.stats(a, b, moments='mvsk')
        # mean=a/(a+b)

        return p1L, p1H, mean


def interpretXYZ(x,y,z,pts=1000): #added
    from scipy.interpolate import griddata
    import random
    if len(x) > 0:
        x, y,z = zip(*random.sample(list(zip(x, y,z)), int(len(x) / 1)))
    xi = np.linspace(min(x), max(x),pts)
    yi = np.linspace(min(y), max(y),pts)
    zi = griddata((x, y), z, (xi[None,:], yi[:,None]), method='cubic')
    return xi,yi,zi

def printCM(matrix):
    """
    print a confusion matrix
    :param matrix:
    :return:
    """
import numpy as np
class ConfusionMatrix(object): #confusion matrix object
    def __init__(self,name="",predictions=["A","NOT(A)"]):
        self.TP=self.TN=self.FP=self.FN=0
        self.name=name
        self.width=30
        self.predictions=predictions
        self.gamesToPredict=[] #list of how many games it took to predict

    @property
    def ntrials(self):
        return self.TP+self.TN+self.FP+self.FN
    @property
    def av_predict(self):
        if sum(self.gamesToPredict)==0:
            av=np.average([-1.0])
        else:
            av=np.average(self.gamesToPredict)
        return av
    @property
    def TPR(self):
        #True Positive Rate Sensitivity
        if self.TP+self.FN==0: return np.nan
        return self.TP/(self.TP+self.FN)
    @property
    def TNR(self):
        #True negative rate TNR Specificity
        if self.TN+self.FP==0: return np.nan

        return self.TN/(self.TN+self.FP)
    @property
    def PPV(self):
        #positive predictive value. Precision
        if self.TP+self.FP==0: return np.nan
        return self.TP/(self.TP+self.FP)
    @property
    def NPV(self):
        #negative predictive value
        if self.TN+self.FN==0: return np.nan
        return self.TN/(self.TN+self.FN)
    @property
    def ACC(self):
        if self.TP+self.TN+self.FP+self.FN==0: return np.nan

        return (self.TP+self.TN)/(self.TP+self.TN+self.FP+self.FN)
    @property
    def F1(self):
        #is the harmonic mean of precision and sensitivity
        try:
            val= 2*(self.PPV*self.TPR)/(self.PPV+self.TPR)
        except:
            val=np.nan
        return val

    @property
    def MCC(self):
        try:
            val=(self.TP*self.TN-self.FP*self.FN)/math.sqrt((self.TP+self.FP)*(self.TP+self.FN)*(self.TN+self.FP)*(self.TN+self.FN))
        except:
            val=np.nan
        return val
    def repEmpty(self):
        line=f"{self.name}________ntrials:{self.ntrials}____Av_predict:{self.av_predict}_______________\n"
        line += '{0: <{width}}'.format("actual   \predicted ->", width=self.width)
        line += "|{0:<7}|{1:<7}".format(self.predictions[0], self.predictions[1])
        line += "\n"

        line += '{0: <{width}}'.format(self.predictions[0], width=self.width)
        line += "|{:<7}|{:<7}".format(np.round(0, 2), np.round(0, 2))
        line += "\n"
        line += '{0: <{width}}'.format(self.predictions[1], width=self.width)
        line += "|{:<7}|{:<7}".format(np.round(0, 2), np.round(0, 2))
        line += "\n"
        return line
    @property
    def allmetrics(self):
        return [self.av_predict,self.TP,self.FN,self.FP,self.TN,self.TPR,self.TNR,self.PPV,self.NPV,self.ACC,self.F1,self.MCC]

    def __repr__(self):
        if self.ntrials==0:
            return self.repEmpty()
        line=f"{self.name}________ntrials:{self.ntrials}______Av_predict:{self.av_predict}_______________________\n"
        line+=f"TPR:{round(self.TPR,2)} TNR:{round(self.TNR,2)} PPV:{round(self.PPV,2)}  NPV:{round(self.NPV,2)}   ACC:{round(self.ACC,2)}  F1:{round(self.F1,2)}   MCC:{round(self.MCC,2)} \n "
        lw = 6
        line += '{0:<{width}}'.format("actual  \predicted ->", width=self.width)
        line += "|{0:<7}|{1:<7}".format(self.predictions[0], self.predictions[1])
        line += "\n"

        line += '{0:<{width}}'.format(self.predictions[0], width=self.width)
        line += "|{:<7}|{:<7}".format(np.round(self.TP / self.ntrials, 2), np.round(self.FN / self.ntrials, 2))
        line +="\n"
        line += '{0:<{width}}'.format(self.predictions[1], width=self.width)
        line += "|{:<7}|{:<7}".format(np.round(self.FP / self.ntrials, 2), np.round(self.TN / self.ntrials, 2))
        line += "\n"
        return line

    def __str__(self):
        return repr(self)
class player(object):
    def __init__(self, pWin):
        self.pWin = pWin
        self.name = str(pWin)
        self.nWins = 0

    def reset(self):
        self.nWins = 0



if __name__ == '__main__':

    p1 = player(0.45)
    p2 = player(0.55)

    g = game(p1, p2,player(0))

    for j in range(100):
        X = 0
        n = 0
        for i in range(4):
            np.random.seed(None)
            winner=g.playGame()
            if winner==p1:
                X+=1
            n+=1
        if X>(n-X):
            print(f"{X},{n}")
