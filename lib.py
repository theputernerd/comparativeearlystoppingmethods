from collections import namedtuple

conf_stopped_struct = namedtuple("conf_struct", "name score p1wins p2wins draws ngames bestPlayerPredict actualBestPlayer predictCorrect time")

import numpy as np

class game(object):
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
