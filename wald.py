import math
from lib import *

"""



n*min{p,1−p}≥10 when can estimate binomial as normal for wald. so if p=0.5 then n=20. 


Just as in a criminal trial, we never conclude that the statement is “innocent” – at most, we find it “not guilty.” In other words, our analysis leaves us in one of two camps: We have strong evidence that the original statement is false, or we do not have such evidence. Therefore, if we wish to make an affirmative case for a claim, we are forced to take the opposite of that claim as the statement we put on trial. Only in this way might we conclude, at the end, that the data – if strong evidence against the claim on trial – serves to support the original claim.
http://www.kellogg.northwestern.edu/faculty/weber/decs-430/decs-430%20session%204/hypothesis_testing.htm

http://sphweb.bumc.bu.edu/otlt/MPH-Modules/BS/BS704_Confidence_Intervals/BS704_Confidence_Intervals_print.html

http://www.statisticshowto.com/probability-and-statistics/confidence-interval/#CIpopprop
It has been known for some time that the Wald interval performs poorly unless n is quite large (e.g., Ghosh 1979, Blyth and Still 1983).
Wald can be prone to unlucky pairs even at high n
Ghosh, B. K. (1979), "A Comparison of Some Approximate Confidence Intervals for the Binomial Parameter," Journcal of the Anmericanz Statistical Association, 74, 894-900
Blyth, C. R., and Still, H. A. (1983), "Binomial Confidence Intervals," Journal of the American Statistical Association, 78, 108-116.
n . min(p,1 - p) is at least 5 (or 10).

"""
class wal_z_score_30min():
    zt = 1.96 #2.576 #1.96 #2.576
    minGames=30
    desc = f"after {minGames} the z-score is calculated; if abs(z) > {zt} then a prediction is made based on" \
           f"whichever player has won the most games. Will only predict a winner, not a draw - ignores drawn games."
    name=f"wald_z_score_30min_zt={zt}"
    finished=False
    @staticmethod
    def reset():
        wal_z_score_30min.finished = False
    @staticmethod
    def start(p1,p2,drawsP,best_actual):
        zt=wal_z_score_30min.zt
        nwins=p1.nWins
        ngames=p1.nWins+p2.nWins
        totGames=  p1.nWins+p2.nWins+drawsP.nWins

        finish = False
        c = None
        z=0
        if wal_z_score_30min.finished:
            return finish, z, c

        if ngames<wal_z_score_30min.minGames:
            return finish, z, c
        #---------------------  Calculate z score and see if it meets threshold
        z=(nwins-ngames/2.0)/math.sqrt(ngames/4.0)

        if z > zt: #zscore predicted game over 95%_1.96, 99%_2.576
            finish=True
            best_predict = 1
        elif z<-zt:
            best_predict = 2
            finish=True
        if finish:
            c=conf_stopped_struct(wal_z_score_30min.name, z, p1.nWins, p2.nWins, drawsP.nWins, totGames, best_predict, best_actual, best_actual == best_predict, -1.0)
            wal_z_score_30min.finished=True
        return finish,z,c

class wal_z_score_100min():
    zt = 1.96 #2.576 #1.96 #2.576
    minGames=100
    desc = f"after {minGames} the z-score is calculated; if abs(z) > {zt} then a prediction is made based on" \
           f"whichever player has won the most games. Will only predict a winner, not a draw - ignores drawn games."
    name=f"wald_z_score_100min_zt={zt}"
    finished=False
    @staticmethod
    def reset():
        wal_z_score_100min.finished = False
    @staticmethod
    def start(p1,p2,drawsP,best_actual):
        zt=wal_z_score_100min.zt
        nwins=p1.nWins
        ngames=p1.nWins+p2.nWins
        totGames=  p1.nWins+p2.nWins+drawsP.nWins

        finish = False
        c = None
        z=0
        if wal_z_score_100min.finished:
            return finish, z, c

        if ngames<wal_z_score_100min.minGames:
            return finish, z, c
        #---------------------  Calculate z score and see if it meets threshold
        z=(nwins-ngames/2.0)/math.sqrt(ngames/4.0)

        if z > zt: #zscore predicted game over 95%_1.96, 99%_2.576
            finish=True
            best_predict = 1
        elif z<-zt:
            best_predict = 2
            finish=True
        if finish:
            c=conf_stopped_struct(wal_z_score_100min.name, z, p1.nWins, p2.nWins, drawsP.nWins, totGames, best_predict, best_actual, best_actual == best_predict, -1.0)
            #if best_actual != best_predict:
            #    print(totGames)
            wal_z_score_100min.finished=True
        return finish,z,c

import scipy.stats as stats

def wald_int(X,n,alpha=0.05):
    z=stats.norm.ppf((1-alpha/2.0)) #two tailed.

    p1hat = float(X) / n
    p1L = p1hat - z * math.sqrt(
        p1hat * (1 - p1hat) / n)  # see http://www.ucl.ac.uk/english-usage/staff/sean/resources/binomialpoisson.pdf eqn1
    p1U = p1hat + z * math.sqrt(p1hat * (1 - p1hat) / n)
    return p1L,p1U

class wal_conf_delta():
    zt = 1.96  # 2.576
    pscore=0.05
    minGames=30
    stoppingPerc=.15 #stops when the delta between upper and low is less than this
    name = f"wald_conf_delta_p={pscore}_conf={stoppingPerc}"
    desc = f"the wald lower and upper confidence bound is calculated for p1 and if the delta is < {stoppingPerc} " \
           f"a prediction is made based on the average of the two bounds. " \
           f"Will only predict a winner."

    finished = False
    @staticmethod
    def reset():
        wal_conf_delta.finished = False
    @staticmethod
    def start(p1,p2,drawsP,best_actual):
        #https://www.evanmiller.org/how-not-to-sort-by-average-rating.html
        nwins = p1.nWins
        ngames = p1.nWins + p2.nWins
        totGames = p1.nWins + p2.nWins + drawsP.nWins
        finished=False
        c = None

        lowerBound=0
        if wal_conf_delta.finished:
            return finished, lowerBound, c

        if ngames<wal_conf_delta.minGames:
            return finished, lowerBound, c
        n=ngames
        if n == 0:
            return finished, lowerBound, c

        #---------------------  Calculate z score and see if it meets threshold
        z = wal_conf_delta.zt  # 1.44 = 85%, 1.96 = 95%
        p1L, p1U=wald_int(p1.nWins,n,wal_conf_delta.pscore)
        deltaP1=p1U-p1L
        #print(str(deltaP1))

        if (deltaP1)<wal_conf_delta.stoppingPerc: #it could converge to a solution lower than 50%
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
            wal_conf_delta.finished=True
            c=conf_stopped_struct(wal_conf_delta.name, [lowerBound, upperBound], p1.nWins, p2.nWins, drawsP.nWins, totGames, best_predict, best_actual, best_actual == best_predict, -1.0)
            if best_actual!=best_predict:
                #print(c)
                pass

        return finished,lowerBound,c

