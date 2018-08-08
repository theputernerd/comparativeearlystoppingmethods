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
def shouldIStop(method,lc,uc,mean,epsilon=0.01): #first number is winner, second number stopping condition.
    #method is needed cause lc and uc should be either 0.025 or 0.05
    ut=np.round(0.5+epsilon,3) #upperthreshold
    lt=np.round(0.5-epsilon,3) #lower threshold
    lc=int(lc*1000)/1000.0#round down. #covers for when python has rounding errors. Floats cause issues at the boundaries.

    uc = np.round(uc, 3)
    mean=np.round(mean,3)

    if method==1:
        if lc>ut:
            return 1,1.1  #player wins from condition 1
        elif uc<lt:
            return 2,1.2 #player loses from condition 1.1
    elif method==2:
        if math.fabs(uc-lc)<epsilon*0.8:#NB that predict some # inside the threshold to account for errors on the edge
            if mean>ut:
                return 1, 2.1  # player wins from condition 2.1
            if mean<lt:
                return 2, 2.2  # player loses from condition 2.2
            if mean>lt and mean<ut:
                return 3, 2.3 #it was a draw
    return 0,0

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
