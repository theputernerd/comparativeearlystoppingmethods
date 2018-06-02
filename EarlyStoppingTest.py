

import numpy as np

class game(object):
    def __init__(self,p1,p2,drawsP):
        self.winner=None
        self.p1=p1
        self.p2=p2
        self.drawsP=drawsP
        assert (p1.pWin+p2.pWin+drawsP.pWin)==1
        pass
        
    def playGame(self):
        winner=np.random.choice([self.p1,self.p2,self.drawsP],p=[self.p1.pWin,self.p2.pWin,self.drawsP.pWin])
        return winner
        
class player(object):
    def __init__(self, pWin):
        self.pWin=pWin
        self.name=str(pWin)
        self.nWins=0
    def reset(self):
        self.nWins=0

import math
from collections import namedtuple

conf_stopped_struct = namedtuple("conf_struct", "name score p1wins p2wins draws ngames bestPlayerPredict actualBestPlayer predictCorrect")
g = 300
class n_games:
    #This test plays g games and chooses a winner after those many games are played based solely on the number of wins.
    #p1>p2 or p2>p1 or p1=p2

    name=f"pure_{g}_games"
    desc=f"plays {g} games and when played makes a prediction based on the number of wins of each player. Predicts either" \
         f"player 1, 2 or 0 for draw. It keeps drawn games in the dataset."
    finished=False
    @staticmethod
    def start(p1,p2,drawsP,best_actual):
        #this condition only plays 300 games then decides based on who has won the most games.
        ngames = p1.nWins + p2.nWins+drawsP.nWins
        finish = False
        c = None
        z = None
        if n_games.finished:  # Already made a prediction
            return finish, z, c

        if ngames<g:
            return finish, z, c

        if (ngames)>g: #needs 300 wins or losses
            return finish, z, c

        finish=True

        if (p1.nWins > p2.nWins):
            best_predict = 1
            finished=True

        elif (p1.nWins < p2.nWins):
            best_predict = 2
            finished=True

        else:
            best_predict =0
            finished=True

        if finished:
            c = conf_stopped_struct(n_games.name, p1.nWins, p1.nWins, p2.nWins, drawsP.nWins, ngames, best_predict, best_actual,best_actual==best_predict)
            n_games.finished=True

        return finish, z, c


class perc_after_n_games:
    name=f"perc_{g}_games"
    finished=False
    threshold=0.55
    desc = f"plays {g} games and makes a prediction based only if a player has won more than {threshold*100} games. Only records wins or losses in the data " \
           f"so the number of actual games may be more than {g} when draws are taken into account. " \
           f"Will not make a prediction if winrate is not above threshold."
    @staticmethod
    def start( p1,p2,drawsP,best_actual):
        #this condition only plays 300 games then decides only if one agent is 55% better.
        finish = False
        c = None
        z = None
        if perc_after_n_games.finished:  # Already made a prediction
            return finish, z, c

        if (p1.nWins + p2.nWins)<g: #needs 300 wins or losses
            return finish, z, c
        if (p1.nWins + p2.nWins)>g: #needs 300 wins or losses
            return finish, z, c

        finish=True
        totGames = p1.nWins + p2.nWins +drawsP.nWins
        nGames=p1.nWins + p2.nWins
        p1winrate=p1.nWins/nGames
        finished=False
        if (p1winrate > perc_after_n_games.threshold):
            best_predict = 1
            finished=True
        elif (p2winrate > perc_after_n_games.threshold):
            best_predict = 2
            finished=True
        else:
            finish=False

        if finished:
            c = conf_stopped_struct(perc_after_n_games.name, p1winrate, p1.nWins, p2.nWins, drawsP.nWins, totGames, best_predict, best_actual,best_actual==best_predict)
            perc_after_n_games.finished=True

        return finish, z, c


class z_score():
    zt = 1.96 #2.576 #1.96 #2.576
    minGames=30
    desc = f"after {minGames} the z-score is calculated; if abs(z) > {zt} then a prediction is made based on" \
           f"whichever player has won the most games. Will only predict a winner, not a draw - ignores drawn games."
    name=f"z_score_zt={zt}"
    finished=False
    @staticmethod
    def start(p1,p2,drawsP,best_actual):
        zt=z_score.zt
        nwins=p1.nWins
        ngames=p1.nWins+p2.nWins
        totGames=  p1.nWins+p2.nWins+drawsP.nWins

        finish = False
        c = None
        z=0
        if z_score.finished:
            return finish, z, c

        if ngames<z_score.minGames:
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
            c=conf_stopped_struct(z_score.name, z,p1.nWins,p2.nWins,drawsP.nWins,totGames,best_predict,best_actual,best_actual==best_predict)
            z_score.finished=True
        return finish,z,c

class wilsonScore():
    zt = 1.96  # 2.576
    name = f"wilson_score_zt={zt}"
    lower_conf_limit=0.50  #this indcates when the test stops and makes a prediction.
    desc = f"the wilson-score lower confidence bound is calculated for both players and if this limit is > {lower_conf_limit} " \
           f"a prediction is made for the player that exceeded the limit." \
           f"Will only predict a winner, not a draw - ignores drawn games."

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
        if wilsonScore.finished:
            return finished, lowerBound, c

        n=ngames
        if n == 0:
            return finished, lowerBound, c

        #---------------------  Calculate z score and see if it meets threshold
        z = wilsonScore.zt  # 1.44 = 85%, 1.96 = 95%
        p1phat = float(p1.nWins) / n
        p1L=((p1phat + z * z / (2 * n) - z * math.sqrt((p1phat * (1 - p1phat) + z * z / (4 * n)) / n)) / (1 + z * z / n))
        p1U=((p1phat + z * z / (2 * n) + z * math.sqrt((p1phat * (1 - p1phat) + z * z / (4 * n)) / n)) / (1 + z * z / n))
        p2phat = float(p2.nWins) / n
        p2L=((p2phat + z * z / (2 * n) - z * math.sqrt((p2phat * (1 - p2phat) + z * z / (4 * n)) / n)) / (1 + z * z / n))
        p2U=((p2phat + z * z / (2 * n) + z * math.sqrt((p2phat * (1 - p2phat) + z * z / (4 * n)) / n)) / (1 + z * z / n))
        ################Now the wilson lower confidence bound is calculated, so now check if is better
        upperBound=0
        if p1L>wilsonScore.lower_conf_limit : #have confidence the lower bound is >0.5
            finished=True
            best_predict=1
            lowerBound=p1L
            upperBound=p1U

        elif p2L>wilsonScore.lower_conf_limit:
            finished=True
            best_predict=2
            lowerBound=p2L
            upperBound=p2U

        if finished:

            c=conf_stopped_struct(wilsonScore.name, [lowerBound,upperBound],p1.nWins,p2.nWins,drawsP.nWins,totGames,best_predict,best_actual,best_actual==best_predict)
            if best_actual!=best_predict:
                #print(c)
                pass
            wilsonScore.finished=True

        return finished,lowerBound,c

class wilsonScore_PercDif():
    zt = 1.96  # 2.576
    stoppingPerc=.1 #stops when the delta between upper and low is less than this
    name = f"wilson_score_zt={zt}_with_bounds_convergence={stoppingPerc}"
    desc = f"the wilson-score lower and upper confidence bound is calculated for p1 and if the delta is < {stoppingPerc} " \
           f"a prediction is made based on the average of the two bounds. " \
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
        if wilsonScore_PercDif.finished:
            return finished, lowerBound, c

        n=ngames
        if n == 0:
            return finished, lowerBound, c

        #---------------------  Calculate z score and see if it meets threshold
        z = wilsonScore_PercDif.zt  # 1.44 = 85%, 1.96 = 95%
        p1phat = float(p1.nWins) / n
        p1L=((p1phat + z * z / (2 * n) - z * math.sqrt((p1phat * (1 - p1phat) + z * z / (4 * n)) / n)) / (1 + z * z / n))
        p1U=((p1phat + z * z / (2 * n) + z * math.sqrt((p1phat * (1 - p1phat) + z * z / (4 * n)) / n)) / (1 + z * z / n))
        p2phat = float(p2.nWins) / n
        p2L=((p2phat + z * z / (2 * n) - z * math.sqrt((p2phat * (1 - p2phat) + z * z / (4 * n)) / n)) / (1 + z * z / n))
        p2U=((p2phat + z * z / (2 * n) + z * math.sqrt((p2phat * (1 - p2phat) + z * z / (4 * n)) / n)) / (1 + z * z / n))
        deltaP1=p1U-p1L
        deltaP2=p2U-p2L
        #print(str(deltaP1))

        if (deltaP1)<wilsonScore_PercDif.stoppingPerc: #it could converge to a solution lower than 50%
            #print(deltaP1)
            finished = True
            if ((p1L+p1U)/2.0)>0.5:
                best_predict = 1
                lowerBound = p1L
                upperBound = p1U
            if ((p1L + p1U) / 2.0) < 0.5:
                best_predict = 2
                lowerBound = p2L
                upperBound = p2U
            if ((p1L + p1U) / 2.0) == 0.5:
                best_predict = 0
                lowerBound = p1L
                upperBound = p1U

        deltaP2=p2U - p2L
        if (deltaP2) < wilsonScore_PercDif.stoppingPerc:
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
                if best_predict!=best_predict2:
                    print(f"didn't agree. predict p1:{best_predict}, {best_predict2}")
            else:
                print(f"p1 didn't give result but p2 did.")
                best_predict=best_predict2

            finished = True


        if finished:
            #print(f"{p1.nWins},{p2.nWins},{lowerBound},{upperBound}")
            c=conf_stopped_struct(wilsonScore_PercDif.name, [lowerBound,upperBound],p1.nWins,p2.nWins,drawsP.nWins,totGames,best_predict,best_actual,best_actual==best_predict)
            if best_actual!=best_predict:
                #print(c)
                pass
            wilsonScore_PercDif.finished=True

        return finished,lowerBound,c




def doTests(tests,p1,p2,drawsP,max_ngames,results):
    p1.reset()
    p2.reset()
    drawsP.reset()
    for t in tests:
        t.finished=False

    if (p1winrate > p2winrate):
        best_actual = 1
    elif (p1winrate < p2winrate):
        best_actual = 2
    else:
        best_actual = 0  # No player is better.

    g=game(p1,p2,drawsP)
    z_complete=False
    ngames_complete=False
    perc_after_n_games_complete=False

    for i in range(1,max_ngames):
        winner= g.playGame()
        #winner = drawsP #change remove me.
        winner.nWins+=1

        for t in tests:  #Do each test
            if not t.finished :
                try:
                    finished, z, c_nGames=t.start(p1,p2,drawsP,best_actual)
                except:
                    pass
                    raise
                if finished:
                    results[c_nGames.name].append(c_nGames)

    return results

if __name__ == '__main__':

    p1winrate=0.59
    p2winrate=0.39
    drawRate=1-(p1winrate+p2winrate)
    assert (p1winrate+p2winrate)<=1

    p1=player(p1winrate)
    p2 = player(p2winrate)
    drawsP=player(drawRate)


    #val=wilsonScore.start(p1,p2,drawsP,1)

    tests=[wilsonScore_PercDif,wilsonScore,z_score,perc_after_n_games,n_games]
    results=dict()
    trialDict=dict()
    for t in tests:
        results[t.name]=[]
        trialDict[t.name]=t


    trials=1000
    for j in range(trials):
        results=doTests(tests,p1,p2,drawsP,max_ngames=500,results=results)
    for key in results:
        print("-------------------------------------------------------------------------------------------------------")
        print(f"                                                  ______{key}______")
        #print(f"{trialDict[key].desc}")
        #print("------------------------")
        predictN=0
        falsePredict=0
        nGamesprediction=[]

        for c in results[key]:
            #now get the percentage of correct predictions
            if c.predictCorrect:
                predictN +=1
            else:
                falsePredict+=1
            nGamesprediction.append(c.ngames)
            avPrediction = np.average(nGamesprediction)

            #print (c)
        avPrediction=np.average(nGamesprediction)
        print (f"avGames_to_predict:{avPrediction:.1f}, incorrect_Predict_rate(type 1):{(falsePredict/trials)*100:.3f}%,failed_to_predict_rate(type2) {(1-len(nGamesprediction)/trials)*100:.3f}%, predicted_n_games:{len(nGamesprediction)},  totalFailure:{(1-predictN/trials)*100:.3f}%")



