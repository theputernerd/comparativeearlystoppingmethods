import math
from lib import *
def AC_z_min_n(p1,p2,drawsP,best_actual,minGames,name):
    zt = ac_z_score_30min.zt
    nwins = p1.nWins
    ngames = p1.nWins + p2.nWins
    totGames = p1.nWins + p2.nWins + drawsP.nWins

    finish = False
    c = None

    if ngames < minGames:
        return finish, 0, c
    # ---------------------  Calculate z score and see if it meets threshold
    zt = ac_z_score_30min.zt
    n = float(ngames)
    if n == 0:  # presumably need this.
        return finish, 0, c
    n_hat = n + zt * zt

    p_hat = 1 / n_hat * (nwins + zt * zt / 2)

    p1L = p_hat - zt * math.sqrt((p_hat / n_hat) * (1 - p_hat))
    p1U = p_hat + zt * math.sqrt((p_hat / n_hat) * (1 - p_hat))
    p0 = 0.5
    p_hat_new = (p1L + p1U) / 2.0
    z = (p_hat_new - p0) / math.sqrt(p0 * (1 - p0) / n)
    # z=(nwins-ngames/2.0)/math.sqrt(ngames/4.0)

    if z > zt:  # zscore predicted game over 95%_1.96, 99%_2.576
        finish = True
        best_predict = 1
    elif z < -zt:
        best_predict = 2
        finish = True
    if finish:
        ac_z_score_30min.finished = True
        c = conf_stopped_struct(name, z, p1.nWins, p2.nWins, drawsP.nWins, totGames, best_predict,
                                best_actual, best_actual == best_predict, -1.0)
    return finish, z, c

class ac_z_score_30min():
    zt = 1.96 #2.576 #1.96 #2.576
    minGames=30
    desc = f"Agresti-coull after {minGames} the z-score is calculated using the mean of the AC confidence bounds and the mean of the hypothesis; " \
           f"if abs(z) > {zt} then a prediction is made based on" \
           f"whichever player has won the most games. Will only predict a winner, not a draw - ignores drawn games."
    name=f"AC_z_score_30min_zt={zt}"
    finished=False
    @staticmethod
    def start(p1,p2,drawsP,best_actual):
        ac_z_score_30min.finished, z, c = AC_z_min_n(p1, p2, drawsP, best_actual,
                                                           ac_z_score_30min.minGames,
                                                           ac_z_score_30min.name)
        return ac_z_score_30min.finished, z, c

class ac_z_score_100min():
    zt = 1.96 #2.576 #1.96 #2.576
    minGames=100
    desc = f"Agresti-coull after {minGames} the z-score is calculated using the mean of the AC confidence bounds and the mean of the hypothesis; " \
           f"if abs(z) > {zt} then a prediction is made based on" \
           f"whichever player has won the most games. Will only predict a winner, not a draw - ignores drawn games."
    name=f"AC_z_score_{100}min_zt={zt}"
    finished=False
    @staticmethod
    def start(p1,p2,drawsP,best_actual):
        ac_z_score_100min.finished, z, c = AC_z_min_n(p1, p2, drawsP, best_actual,
                                                      ac_z_score_100min.minGames,
                                                      ac_z_score_100min.name)
        return ac_z_score_100min.finished, z, c
