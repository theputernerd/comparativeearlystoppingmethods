import math
from lib import *
"""
The cost of using exact condence intervals for a binomial proportion Mans Thulin Department of Mathematics, Uppsala University
Theorum 1.
However, this procedure is necessarily conservative: Article Approximate Is Better than "Exact" for Interval Estimation of Binomial Proportions Agresti, Alan ; Coull, Brent A. The American Statistician, 1 May 1998, Vol.52(2), pp.119-126

"""


class clopper_pearson_mean_conf():
    zt = 1.96  # 2.576
    stoppingPerc=.1 #stops when the delta between upper and low is less than this
    name = f"CP_mean_conf_z={zt}_conf={stoppingPerc}"
    desc = f"Cp interval found < {stoppingPerc} " \
           f"delta upper and lower is stopping condition. . " \
           f"Will only predict a winner."

    finished = False
    @staticmethod
    def start(p1,p2,drawsP,best_actual):
        #https://www.evanmiller.org/how-not-to-sort-by-average-rating.html
        nwins = p1.nWins
        ngames = p1.nWins + p2.nWins
        totGames = p1.nWins + p2.nWins + drawsP.nWins
        finished=False
        c = None
        lowerBound=0
        if clopper_pearson_mean_conf.finished:
            return finished, lowerBound, c

        n=float(ngames)
        if n == 0:
            return finished, lowerBound, c

        #---------------------  Calculate z score and see if it meets threshold
        z = clopper_pearson_mean_conf.zt  # 1.44 = 85%, 1.96 = 95%
        p1hat = float(p1.nWins) / n
        p1L=p1hat-(1.0/math.sqrt(n))*z*math.sqrt(p1hat*(1-p1hat))+((2.0*(0.5-p1hat)*z*z-(1+p1hat)))/(3*n)
        p1U=p1hat+(1.0/math.sqrt(n))*z*math.sqrt(p1hat*(1-p1hat))+((2.0*(0.5-p1hat)*z*z+(1+p1hat)))/(3*n)
        #((p1hat + z * z / (2 * n) - z * math.sqrt((p1hat * (1 - p1hat) + z * z / (4 * n)) / n)) / (1 + z * z / n))
        #p1U=((p1hat + z * z / (2 * n) + z * math.sqrt((p1hat * (1 - p1hat) + z * z / (4 * n)) / n)) / (1 + z * z / n))

        deltaP1=p1U-p1L
        #print(str(deltaP1))

        if (deltaP1)<clopper_pearson_mean_conf.stoppingPerc: #it could converge to a solution lower than 50%
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
            clopper_pearson_mean_conf.finished=True
            c=conf_stopped_struct(clopper_pearson_mean_conf.name, [lowerBound, upperBound], p1.nWins, p2.nWins, drawsP.nWins, totGames, best_predict, best_actual, best_actual == best_predict, -1.0)
            if best_actual!=best_predict:
                #print(c)
                pass

        return finished,lowerBound,c

