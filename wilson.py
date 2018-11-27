import math
from lib import *
"""
https://www.mwsug.org/proceedings/2008/pharma/MWSUG-2008-P08.pdf
https://www.evanmiller.org/how-not-to-sort-by-average-rating.html
https://math.stackexchange.com/questions/718279/explanation-for-the-wilson-score-interval
http://www.ucl.ac.uk/english-usage/staff/sean/resources/binomialpoisson.pdf
"This interval has good properties even for a small number of trials and/or an extreme probability" 
https://en.wikipedia.org/wiki/Binomial_proportion_confidence_interval
"""
import scipy.stats as stats
def wils_int(X,n,alpha,cc=True): ##cc==True means with continuity correction.
        #z_calced=abs(stats.norm.ppf((1-alpha/2.0))) #two tailed.
        #global z
        #z = abs(stats.norm.ppf((1 - alpha / 2.0)))  # 1.96 for alpha=0.5
        z = abs(stats.norm.ppf((1 - alpha)))

        if n==0:
            return 0,1,0.5
        p_hat = float(X) / n
        q=1-p_hat

        if cc:
            try:
                if p_hat == 0:
                    p1L=0
                else:
                    p1L = max(0,(2 * n * p_hat + z * z - (z * math.sqrt(z * z - 1 / n + 4 * n * p_hat * (1 - p_hat) + (4 * p_hat - 2)) + 1)) / (2 * (n + z * z)))
                if p_hat==1:
                    p1U=1
                else:
                    p1U = min(1,(2 * n * p_hat + z * z + (z * math.sqrt(z * z - 1 / n + 4 * n * p_hat * (1 - p_hat) - (4 * p_hat - 2)) + 1)) / (2 * (n + z * z)))

            except ValueError:
                print(f"X:{X},n:{n},p_hat:{p_hat},z:{z}")
                raise
        else:
            p1L = ((p_hat + z * z / (2 * n) - z * math.sqrt((p_hat * (1 - p_hat) + z * z / (4 * n)) / n)) / (1 + z * z / n))
            p1U = ((p_hat + z * z / (2 * n) + z * math.sqrt((p_hat * (1 - p_hat) + z * z / (4 * n)) / n)) / (1 + z * z / n))
        E=n*p_hat
        mean=(p1U+p1L)/2.0
        return p1L,p1U,mean

class wilson_lcb():
    global z
    zt = z # 2.576
    name = f"wils_lcb_z=={zt}"
    lower_conf_limit=0.50  #this indcates when the test stops and makes a prediction.
    desc = f"the wilson-probability lower confidence bound is calculated for both players and if LCB is > {lower_conf_limit} " \
           f"a prediction is made for the player that exceeded the limit." \
           f"Will only predict a winner, not a draw - ignores drawn games."

    finished = False

    @staticmethod
    def reset():
        wilson_lcb.finished = False
    @staticmethod
    def start(p1,p2,drawsP,best_actual):
        #https://www.evanmiller.org/how-not-to-sort-by-average-rating.html
        nwins = p1.nWins
        ngames = p1.nWins + p2.nWins
        totGames = p1.nWins + p2.nWins + drawsP.nWins
        finished=False
        c = None
        lowerBound=0
        if wilson_lcb.finished:
            return finished, lowerBound, c

        n=ngames
        if n == 0:
            return finished, lowerBound, c

        #---------------------  Calculate z score and see if it meets threshold
        z_l = wilson_lcb.zt  # 1.44 = 85%, 1.96 = 95%
        p1phat = float(p1.nWins) / n
        p1L=((p1phat + z_l * z_l / (2 * n) - z_l * math.sqrt((p1phat * (1 - p1phat) + z_l * z_l / (4 * n)) / n)) / (1 + z_l * z_l / n))
        p1U=((p1phat + z_l * z_l / (2 * n) + z_l * math.sqrt((p1phat * (1 - p1phat) + z_l * z_l / (4 * n)) / n)) / (1 + z_l * z_l / n))
        p2phat = float(p2.nWins) / n
        p2L=((p2phat + z_l * z_l / (2 * n) - z_l * math.sqrt((p2phat * (1 - p2phat) + z_l * z_l / (4 * n)) / n)) / (1 + z_l * z_l / n))
        p2U=((p2phat + z_l * z_l / (2 * n) + z_l * math.sqrt((p2phat * (1 - p2phat) + z_l * z_l / (4 * n)) / n)) / (1 + z_l * z_l / n))
        ################Now the wilson lower confidence bound is calculated, so now check if is better
        upperBound=0
        if p1L>wilson_lcb.lower_conf_limit : #have confidence the lower bound is >0.5
            finished=True
            best_predict=1
            lowerBound=p1L
            upperBound=p1U

        elif p2L>wilson_lcb.lower_conf_limit:
            finished=True
            best_predict=2
            lowerBound=p2L
            upperBound=p2U

        if finished:

            c=conf_stopped_struct(wilson_lcb.name, [lowerBound, upperBound], p1.nWins, p2.nWins, drawsP.nWins, totGames, best_predict, best_actual, best_actual == best_predict, -1.0)
            if best_actual!=best_predict:
                #print(c)
                pass
            wilson_lcb.finished=True

        return finished,lowerBound,c


class wilson_conf_delta():
    global z
    zt = z  # 2.576
    stoppingPerc=.15 #stops when the delta between upper and low is less than this
    name = f"wils_CC_conf_delta_z={zt}_conf={stoppingPerc}"
    desc = f"the wilson-score lower and upper confidence bound is calculated for p1 and if the delta is < {stoppingPerc} " \
           f"a prediction is made based on the average of the two bounds. " \
           f"Will only predict a winner."

    finished = False

    @staticmethod
    def reset():
        wilson_conf_delta.finished = False

    @staticmethod
    def start(p1,p2,drawsP,best_actual):
        #https://www.evanmiller.org/how-not-to-sort-by-average-rating.html
        nwins = p1.nWins
        ngames = p1.nWins + p2.nWins
        totGames = p1.nWins + p2.nWins + drawsP.nWins
        finished=False
        c = None
        lowerBound=0
        if wilson_conf_delta.finished:
            return finished, lowerBound, c

        n=ngames
        if n == 0:
            return finished, lowerBound, c
        global alpha
        #---------------------  Calculate z score and see if it meets threshold
        zl = wilson_conf_delta.zt  # 1.44 = 85%, 1.96 = 95%
        p1L, p1U,mean =wils_int(p1.nWins,n,alpha)

        deltaP1=abs(p1U-p1L)
        #print(str(deltaP1))
        #print(deltaP1)
        if (deltaP1)<wilson_conf_delta.stoppingPerc: #it could converge to a solution lower than 50%
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
        ################ Not neccesary to do the other player but code is here just in case you want to have a look
        ######


        if finished:
            #print(f"{p1.nWins},{p2.nWins},{lowerBound},{upperBound}")
            wilson_conf_delta.finished=True
            c=conf_stopped_struct(wilson_conf_delta.name, [lowerBound, upperBound], p1.nWins, p2.nWins, drawsP.nWins, totGames, best_predict, best_actual, best_actual == best_predict, -1.0)
            if best_actual!=best_predict:
                #print(c)
                pass

        return finished,lowerBound,c

def wilson_z_score_No_CC_n_min(p1,p2,drawsP,best_actual,minGames,name):
        global z
        zt = z


        ngames = p1.nWins + p2.nWins
        totGames = p1.nWins + p2.nWins + drawsP.nWins

        finish = False
        c = None


        if ngames < minGames:
            return finish, 0, c
        # ---------------------  Calculate z score and see if it meets threshold
        n = float(ngames)
        if n == 0:
            return finish, 0, c

        p_hat = float(p1.nWins) / n
        #p1L,p1U=wils_int(p1.nWins,n,0.5,cc=True)
        p1L,p1U,mean=wils_int(p1.nWins,n,cc=False)

        #p1L = (2*n*p_hat+zt*zt-(zt*math.sqrt(zt*zt-1/n+4*n*p_hat*(1-p_hat)+(4*p_hat-2))+1))/(2*(n+zt*zt))
        #p1U = (2*n*p_hat+zt*zt+(zt*math.sqrt(zt*zt-1/n+4*n*p_hat*(1-p_hat)+(4*p_hat-2))+1))/(2*(n+zt*zt))

        p0 = 0.5
        p_hat_new = (p1L + p1U) / 2.0
        zl = (p_hat_new - p0) / math.sqrt(p0 * (1 - p0) / n)
        # z=(nwins-ngames/2.0)/math.sqrt(ngames/4.0)

        if zl > zt:  # zscore predicted game over 95%_1.96, 99%_2.576
            finish = True
            best_predict = 1
        elif zl < -zt:
            best_predict = 2
            finish = True
        if finish:
            c = conf_stopped_struct(name, zl, p1.nWins, p2.nWins, drawsP.nWins, totGames,
                                    best_predict, best_actual, best_actual == best_predict, -1.0)
        return finish, zl, c

class wilsonNOCC_z_score_30min():
    global z
    zt = z  # 2.576 #1.96 #2.576
    minGames = 30
    desc = f"Wil without cont correction. after {minGames} the z-score is calculated using the mean of the wilson confidence bounds and the mean of the hypothesis; " \
           f"if abs(z) > {zt} then a prediction is made based on" \
           f"whichever player has won the most games. Will only predict a winner, not a draw - ignores drawn games."
    name = f"wils_NO_CC_z_score_{minGames}min_zt={zt}"
    finished = False
    @staticmethod
    def reset():
        wilsonNOCC_z_score_30min.finished = False
    @staticmethod
    def start(p1, p2, drawsP, best_actual):
        if wilsonNOCC_z_score_30min.finished:
            return True, 0, None

        wilsonNOCC_z_score_30min.finished, z, c= wilson_z_score_No_CC_n_min(p1, p2, drawsP, best_actual, wilsonNOCC_z_score_30min.minGames, wilsonNOCC_z_score_30min.name)

        return wilsonNOCC_z_score_30min.finished, z, c
class wilsonNOCC_z_score_100min():
    global z
    zt = z  # 2.576 #1.96 #2.576
    minGames = 100
    desc = f"Wil with continuity correction. after {minGames} the z-score is calculated using the mean of the wilson confidence bounds and the mean of the hypothesis; " \
           f"if abs(z) > {zt} then a prediction is made based on" \
           f"whichever player has won the most games. Will only predict a winner, not a draw - ignores drawn games."
    name = f"wils_NO_CC_z_score_{minGames}min_zt={zt}"
    finished = False
    @staticmethod
    def reset():
        wilsonNOCC_z_score_100min.finished = False
    @staticmethod
    def start(p1, p2, drawsP, best_actual):
        if wilsonNOCC_z_score_100min.finished:
            return True, 0, None

        wilsonNOCC_z_score_100min.finished, z, c= wilson_z_score_No_CC_n_min(p1, p2, drawsP, best_actual, wilsonNOCC_z_score_100min.minGames, wilsonNOCC_z_score_100min.name)

        return wilsonNOCC_z_score_100min.finished, z, c


def wilson_z_score_n_min(p1, p2, drawsP, best_actual, minGames, name):
    global z
    global alpha
    zt = z
    retn_val=None
    ngames = p1.nWins + p2.nWins
    totGames = p1.nWins + p2.nWins + drawsP.nWins

    finish = False
    c = None

    if ngames < minGames:
        return finish, 0, c
    # ---------------------  Calculate z score and see if it meets threshold
    n = float(ngames)
    if n == 0:
        return finish, 0, c


    p_hat = float(p1.nWins) / n
    p1L,p1U,mean=wils_int(p1.nWins,n,cc=True,alpha=0.05)
    p2L,p2U,mean2=wils_int(p2.nWins,n,cc=True,alpha=0.05)

    pred1=LCBTest(p1L, p1U)

    pred2=DeltaTest(p1L, p1U)

    if pred1: #then LCB>0.5
        finish=True
        best_predict=1 if p1.nWins>p2.nWins else 2
        retn_val="pred1",p1L,p1U
        if (best_actual != best_predict):
            print("wils {pred1} error")
            print(f"{p1L}-{p1U},{p1.nWins}, {p2.nWins}, {drawsP.nWins}")

    if pred2:
        retn_val = "pred2",p1L, p1U
        best_predict=1 if p1.nWins>p2.nWins else 2 ###TODO: WHAT ABOUT AROUND 0.5 then its a draw.
        finish = True
        if (best_actual != best_predict):
            print("wils {pred2} error")
            print(f"{p1L}-{p1U},{p1.nWins}, {p2.nWins}, {drawsP.nWins}")

        if pred1:
            print("Both tests demand stopping in Wilson")

    if finish:
        c = conf_stopped_struct(name, retn_val, p1.nWins, p2.nWins, drawsP.nWins, totGames,
                                best_predict, best_actual, best_actual == best_predict, -1.0)
    return finish, retn_val, c

class wilson_z_score_0min():
    global z
    zt = z  # 2.576 #1.96 #2.576
    minGames = 1
    desc = f"after {minGames} the z-score is calculated using the mean of the wilson confidence bounds and the mean of the hypothesis; " \
           f"if abs(z) > {zt} then a prediction is made based on" \
           f"whichever player has won the most games. Will only predict a winner, not a draw - ignores drawn games."
    name = f"wils_z_scoreCC_{minGames}min_zt={zt}"
    finished = False
    @staticmethod
    def reset():
        wilson_z_score_0min.finished = False
    @staticmethod
    def start(p1, p2, drawsP, best_actual):
        zr=wilson_z_score_0min.zt
        if wilson_z_score_0min.finished:
            return True, 0, None

        wilson_z_score_0min.finished,zr,c= wilson_z_score_n_min(p1,p2,drawsP,best_actual,wilson_z_score_0min.minGames,wilson_z_score_0min.name)

        return wilson_z_score_0min.finished,zr,c

class wilson_z_score_30min():
    global z
    zt = z  # 2.576 #1.96 #2.576
    minGames = 30
    desc = f"after {minGames} the z-score is calculated using the mean of the wilson confidence bounds and the mean of the hypothesis; " \
           f"if abs(z) > {zt} then a prediction is made based on" \
           f"whichever player has won the most games. Will only predict a winner, not a draw - ignores drawn games."
    name = f"wils_z_scoreCC_{minGames}min_zt={zt}"
    finished = False
    @staticmethod
    def reset():
        wilson_z_score_30min.finished = False
    @staticmethod
    def start(p1, p2, drawsP, best_actual):
        if wilson_z_score_30min.finished:
            return True, 0, None

        wilson_z_score_30min.finished,z,c= wilson_z_score_n_min(p1,p2,drawsP,best_actual,wilson_z_score_30min.minGames,wilson_z_score_30min.name)

        return wilson_z_score_30min.finished,z,c


class wilson_z_score_100min():
    global z
    zt = z #2.576 #1.96 #2.576
    minGames=100
    desc = f"after {minGames} the z-score is calculated using the mean of the wilson confidence bounds and the mean of the hypothesis; " \
           f"if abs(z) > {zt} then a prediction is made based on" \
           f"whichever player has won the most games. Will only predict a winner, not a draw - ignores drawn games."
    name=f"wils_z_scoreCC_{minGames}min_zt={zt}"
    finished=False
    @staticmethod
    def reset():
        wilson_z_score_100min.finished = False
    @staticmethod
    def start(p1,p2,drawsP,best_actual):
        if wilson_z_score_100min.finished:
            return True, 0, None
        wilson_z_score_100min.finished,z,c=wilson_z_score_n_min(p1,p2,drawsP,best_actual,wilson_z_score_100min.minGames,wilson_z_score_100min.name)
        return wilson_z_score_100min.finished,z,c

