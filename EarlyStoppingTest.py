

"""
http://www.ucl.ac.uk/english-usage/staff/sean/resources/binomialpoisson.pdf noet figure 6 showing wilson/wald difference with n=100

Type I errors are like “false positives” and happen when you conclude that the variation you’re experimenting with is a “winner” when it’s actually not. Scientifically, this means that you are incorrectly rejecting the true null hypothesis and believe a relationship exists when it actually doesn’t. The chance that you commit type I errors is known as the type I error rate or significance level (p-value)--this number is conventionally and arbitrarily set to 0.05 (5%).
Type II errors are like “false negatives,” an incorrect rejection that a variation in a test has made no statistically significant difference. Statistically speaking, this means you’re mistakenly believing the false null hypothesis and think a relationship doesn’t exist when it actually does. You commit a type 2 error when you don’t believe something that is in fact true.
https://www.optimizely.com/anz/optimization-glossary/type-2-error/#


Article Approximate Is Better than "Exact" for Interval Estimation of Binomial Proportions Agresti, Alan ; Coull, Brent A. The American Statistician, 1 May 1998, Vol.52(2), pp.119-126

Sample size for estimating a binomial proportion: comparison of different methods Luzia Gonçalves , M. Rosário de Oliveira , Cláudia Pascoal  & Ana Pires
"""
from wilson import wilson_lcb,wilson_conf_delta,wilson_z_score_100min,wilson_z_score_30min,wilsonCC_z_score_30min,wilsonCC_z_score_100min
from wald import wal_z_score_30min,wal_z_score_100min, wal_conf_delta
from naiveMethods import perc_after_n_games,n_games
from timeit import default_timer as timer
from clopper_pearson import clopper_pearson_mean_conf
from agresti_coull import ac_z_score_30min,ac_z_score_100min
from lib import *

def doTests(tests,p1,p2,drawsP,max_ngames,results):
    p1.reset()
    p2.reset()
    drawsP.reset()

    for t in tests:
        t.reset()
    #bayes.reset() #it's the only one with reset global variables.

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
                    t0=timer()
                    finished, z, c_nGames=t.start(p1,p2,drawsP,best_actual)
                    t1=timer()
                except:
                    pass
                    raise
                if finished:
                    timetaken=t1-t0

                    c_nGames=c_nGames._replace(time=timetaken)
                    results[c_nGames.name].append(c_nGames)

    return results
from lib import *
import time
from bayes import bayes
if __name__ == '__main__':

    p1winrate=0.45
    p2winrate=0.550
    trials=100
    maxNgames=500

    drawRate=1-(p1winrate+p2winrate)
    assert (p1winrate+p2winrate)<=1

    p1=player(p1winrate)
    p2 = player(p2winrate)
    drawsP=player(drawRate)
    #val=wilsonScore.start(p1,p2,drawsP,1)

    tests=[bayes,wilson_lcb, clopper_pearson_mean_conf, wilson_conf_delta, wal_conf_delta, ac_z_score_30min,wilson_z_score_30min,wilsonCC_z_score_30min, wal_z_score_30min, ac_z_score_100min,wilson_z_score_100min,wilsonCC_z_score_100min, wal_z_score_100min, perc_after_n_games, n_games]
    results=dict()
    trialDict=dict()
    for t in tests:
        results[t.name]=[]
        trialDict[t.name]=t

    times={}
    for j in range(trials):
        t0=time.time()
        results=doTests(tests,p1,p2,drawsP,max_ngames=maxNgames,results=results)

        t1=time.time()
    str = ""
    maxlenName=0
    for key in results: #This get the max so the columns are aligned.

        if len(key)>maxlenName:
            maxlenName=len(key)

    for key in results:
        #print(f"{trialDict[key].desc}")
        #print("------------------------")
        predictN=0
        falsePredict=0
        nGamesprediction=[]
        timetaken=[]
        if len(key)>maxlenName:
            maxlenName=len(key)
        for c in results[key]:

            #now get the percentage of correct predictions
            if c.predictCorrect:
                predictN +=1
            else:
                falsePredict+=1
            nGamesprediction.append(c.ngames)
            #avPrediction = np.average(nGamesprediction)
            timetaken.append(c.time)
            #print (c)
        avTime = np.average(timetaken)
        avPrediction=np.average(nGamesprediction)
        #print("-------------------------------------------------------------------------------------------------------")
        #print(f"                                                  ______{key}______ av_time:{avTime:1.2E}s")

        str+=f"{key}"+" "*(maxlenName-len(key))+f"|{avPrediction:.1f}\t\t|\t{(falsePredict/trials)*100:.3f}%  \t|\t{(1-len(nGamesprediction)/trials)*100:.3f}%\n"#  \t|\t{len(nGamesprediction)}\t\t\n"
        #print (f"avGames_to_predict:{avPrediction:.1f}, incorrect_Predict_rate(type 1):{(falsePredict/trials)*100:.3f}%,failed_to_predict_rate(type2) {(1-len(nGamesprediction)/trials)*100:.3f}%, predicted_n_games:{len(nGamesprediction)},  totalFailure:{(1-predictN/trials)*100:.3f}%")
    strline1 = "Name"+" "*(maxlenName-4)+f"|\t n_games \t| Type 1 E  \t|\tType2 E \t"#  \t| npredict  \t"
    print(f"trials{trials}, max_n_games{maxNgames}")
    print(strline1)
    print(str)



