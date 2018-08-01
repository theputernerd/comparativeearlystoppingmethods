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
