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
class wilson_lcb():
    zt = 1.96  # 2.576
    name = f"wil_lcb_z=={zt}"
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
        z = wilson_lcb.zt  # 1.44 = 85%, 1.96 = 95%
        p1phat = float(p1.nWins) / n
        p1L=((p1phat + z * z / (2 * n) - z * math.sqrt((p1phat * (1 - p1phat) + z * z / (4 * n)) / n)) / (1 + z * z / n))
        p1U=((p1phat + z * z / (2 * n) + z * math.sqrt((p1phat * (1 - p1phat) + z * z / (4 * n)) / n)) / (1 + z * z / n))
        p2phat = float(p2.nWins) / n
        p2L=((p2phat + z * z / (2 * n) - z * math.sqrt((p2phat * (1 - p2phat) + z * z / (4 * n)) / n)) / (1 + z * z / n))
        p2U=((p2phat + z * z / (2 * n) + z * math.sqrt((p2phat * (1 - p2phat) + z * z / (4 * n)) / n)) / (1 + z * z / n))
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
    zt = 1.96  # 2.576
    stoppingPerc=.1 #stops when the delta between upper and low is less than this
    name = f"wil_conf_delta_z={zt}_conf={stoppingPerc}"
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

        #---------------------  Calculate z score and see if it meets threshold
        z = wilson_conf_delta.zt  # 1.44 = 85%, 1.96 = 95%
        p1hat = float(p1.nWins) / n
        p1L=((p1hat + z * z / (2 * n) - z * math.sqrt((p1hat * (1 - p1hat) + z * z / (4 * n)) / n)) / (1 + z * z / n))
        p1U=((p1hat + z * z / (2 * n) + z * math.sqrt((p1hat * (1 - p1hat) + z * z / (4 * n)) / n)) / (1 + z * z / n))

        deltaP1=p1U-p1L
        #print(str(deltaP1))

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
        """
        p2hat = float(p2.nWins) / n
        p2L=((p2hat + z * z / (2 * n) - z * math.sqrt((p2hat * (1 - p2hat) + z * z / (4 * n)) / n)) / (1 + z * z / n))
        p2U=((p2hat + z * z / (2 * n) + z * math.sqrt((p2hat * (1 - p2hat) + z * z / (4 * n)) / n)) / (1 + z * z / n))
        deltaP2=p2U-p2L
        if (deltaP2) < wilson_mean_conf.stoppingPerc:
            print('p2 found solution')
            #print(deltaP1)

            if ((p2L + p2U) / 2.0) > 0.5:
                best_predict2 = 2
                lowerBound = p2L
                upperBound = p2U
            if ((p2L + p2U) / 2.0) < 0.5:
                best_predict2 = 1
                lowerBound = p2L
                upperBound = p2U
            if ((p2L + p2U) / 2.0) == 0.5:
                best_predict2 = 0
                lowerBound = p2L
                upperBound = p2U

            if finished==True:
                #the other side made a prediction as well.
                print("Both sides made prediction at same time.")
                if best_predict!=best_predict2:
                    print(f"didn't agree. predict p1:{best_predict}, {best_predict2}")
            else:
                print(f"p1 didn't give result but p2 did.")
                best_predict=best_predict2

            finished = True
        """

        if finished:
            #print(f"{p1.nWins},{p2.nWins},{lowerBound},{upperBound}")
            wilson_conf_delta.finished=True
            c=conf_stopped_struct(wilson_conf_delta.name, [lowerBound, upperBound], p1.nWins, p2.nWins, drawsP.nWins, totGames, best_predict, best_actual, best_actual == best_predict, -1.0)
            if best_actual!=best_predict:
                #print(c)
                pass

        return finished,lowerBound,c

def wilson_z_score_CC_n_min(p1,p2,drawsP,best_actual,minGames,name):
        zt = 1.96


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
        p1L = (2*n*p_hat+zt*zt-(zt*math.sqrt(zt*zt-1/n+4*n*p_hat*(1-p_hat)+(4*p_hat-2))+1))/(2*(n+zt*zt))
        p1U = (2*n*p_hat+zt*zt+(zt*math.sqrt(zt*zt-1/n+4*n*p_hat*(1-p_hat)+(4*p_hat-2))+1))/(2*(n+zt*zt))

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
            c = conf_stopped_struct(name, z, p1.nWins, p2.nWins, drawsP.nWins, totGames,
                                    best_predict, best_actual, best_actual == best_predict, -1.0)
        return finish, z, c

class wilsonCC_z_score_30min():
    zt = 1.96  # 2.576 #1.96 #2.576
    minGames = 30
    desc = f"Wil with continuity correction. after {minGames} the z-score is calculated using the mean of the wilson confidence bounds and the mean of the hypothesis; " \
           f"if abs(z) > {zt} then a prediction is made based on" \
           f"whichever player has won the most games. Will only predict a winner, not a draw - ignores drawn games."
    name = f"wilsCC_z_score_{minGames}min_zt={zt}"
    finished = False
    @staticmethod
    def reset():
        wilsonCC_z_score_30min.finished = False
    @staticmethod
    def start(p1, p2, drawsP, best_actual):
        if wilsonCC_z_score_30min.finished:
            return True, 0, None

        wilsonCC_z_score_30min.finished,z,c= wilson_z_score_CC_n_min(p1,p2,drawsP,best_actual,wilsonCC_z_score_30min.minGames,wilsonCC_z_score_30min.name)

        return wilsonCC_z_score_30min.finished,z,c
class wilsonCC_z_score_100min():
    zt = 1.96  # 2.576 #1.96 #2.576
    minGames = 100
    desc = f"Wil with continuity correction. after {minGames} the z-score is calculated using the mean of the wilson confidence bounds and the mean of the hypothesis; " \
           f"if abs(z) > {zt} then a prediction is made based on" \
           f"whichever player has won the most games. Will only predict a winner, not a draw - ignores drawn games."
    name = f"wilsCC_z_score_{minGames}min_zt={zt}"
    finished = False
    @staticmethod
    def reset():
        wilsonCC_z_score_100min.finished = False
    @staticmethod
    def start(p1, p2, drawsP, best_actual):
        if wilsonCC_z_score_100min.finished:
            return True, 0, None

        wilsonCC_z_score_100min.finished,z,c= wilson_z_score_CC_n_min(p1,p2,drawsP,best_actual,wilsonCC_z_score_100min.minGames,wilsonCC_z_score_100min.name)

        return wilsonCC_z_score_100min.finished,z,c


def wilson_z_score_n_min(p1, p2, drawsP, best_actual, minGames, name):
    zt = 1.96

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
    p1L = ((p_hat + zt * zt / (2 * n) - zt * math.sqrt((p_hat * (1 - p_hat) + zt * zt / (4 * n)) / n)) / (
            1 + zt * zt / n))
    p1U = ((p_hat + zt * zt / (2 * n) + zt * math.sqrt((p_hat * (1 - p_hat) + zt * zt / (4 * n)) / n)) / (
            1 + zt * zt / n))
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
        c = conf_stopped_struct(name, z, p1.nWins, p2.nWins, drawsP.nWins, totGames,
                                best_predict, best_actual, best_actual == best_predict, -1.0)
    return finish, z, c


class wilson_z_score_30min():
    zt = 1.96  # 2.576 #1.96 #2.576
    minGames = 30
    desc = f"after {minGames} the z-score is calculated using the mean of the wilson confidence bounds and the mean of the hypothesis; " \
           f"if abs(z) > {zt} then a prediction is made based on" \
           f"whichever player has won the most games. Will only predict a winner, not a draw - ignores drawn games."
    name = f"wil_z_score_{minGames}min_zt={zt}"
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
    zt = 1.96 #2.576 #1.96 #2.576
    minGames=100
    desc = f"after {minGames} the z-score is calculated using the mean of the wilson confidence bounds and the mean of the hypothesis; " \
           f"if abs(z) > {zt} then a prediction is made based on" \
           f"whichever player has won the most games. Will only predict a winner, not a draw - ignores drawn games."
    name=f"wil_z_score_{minGames}min_zt={zt}"
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

