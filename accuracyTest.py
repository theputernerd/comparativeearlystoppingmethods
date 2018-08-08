
import matplotlib as mpl
mpl.use('Agg')
import numpy as np
import random
from lib import player
playerPoolDist = random.uniform(1.5, 1.9)
from wilson import wils_int
from lib import shouldIStop,game
from bayes import bayesian_U
tests = [bayesian_U, wils_int]
import time

import matplotlib.pyplot as plt

def doTests(tests,p1,p2,drawsP,max_ngames,results,epsilon=0.01):
    p1.reset()
    p2.reset()
    drawsP.reset()

    for t in tests:
        t.reset()
    #bayes.reset() #it's the only one with reset global variables.
    ut=0.5+epsilon #upperthreshold
    lt=0.5-epsilon #lower threshold
    if (p1.pWin > ut): #determine the actual winner
        best_actual = 1
    elif (p1.pWin < lt):
        best_actual = 2
    else:
        best_actual = 3  # No player is better.

    g=game(p1,p2,drawsP)

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

def playOneGame(g, results):
    winner = g.playGame()
    # winner = drawsP #change remove me.
    winner.nWins += 1
    return results

import copy
def playGames(p1winrate,p2winrate=None,drawRate=None,trials=1,epsilon=0.01,pmargin=0.9):

    results=dict()
    if p2winrate==None:
        p2winrate = 1 - p1winrate  # probability of p2 winning
    if drawRate==None:
        drawRate = 1 - (p1winrate + p2winrate)
    assert (p1winrate + p2winrate) <= 1
    p1 = player(p1winrate)
    p2 = player(p2winrate)
    drawsP = player(drawRate)

    ut=0.5+epsilon #upperthreshold
    lt=0.5-epsilon #lower threshold
    if (p1.pWin > ut): #determine the actual better player
        best_actual = 1
    elif (p1.pWin < lt):
        best_actual = 2
    else:
        best_actual = 3


    wilsPredicted=False
    baysPredicted=False
    g = game(p1, p2, drawsP)

    wilsonresults = []
    bayesresults=[]
    p1.reset()
    p2.reset()
    drawsP.reset()
    Wcondition1=[]
    Wcondition2 = []
    Bcondition1=[]
    Bcondition2 = []

    while not(wilsPredicted and baysPredicted): # keep going until both methods made a prediction
        #play one game
        results=playOneGame(g,results) #NB results not used
        n=p1.nWins+p2.nWins+drawsP.nWins

        if not wilsPredicted:
            p1L, p1U,mean =wils_int(p1.nWins,n,0.05)
            winner,method=shouldIStop(1,p1L, p1U, mean,epsilon=epsilon,pmargin=pmargin)
            if winner!=0:  ##condition1
                wilsPredicted=True
                # now to see if prediction is correct.
                if int(method) != int(1):
                    assert False
                if winner == 0:
                    assert False  # should have made a prediction.
                else:
                    if winner == best_actual:
                        # corrrect
                        storedResult = [p1.pWin, n, True, method, p1L, p1U,mean,winner,best_actual]
                        wilsonresults=storedResult
                        Wcondition1=storedResult
                    else:
                        storedResult = [p1.pWin, n, False, method, p1L, p1U, mean, winner,best_actual]
                        wilsonresults = storedResult
                        Wcondition1 = storedResult
            p1L, p1U, mean = wils_int(p1.nWins, n, 0.025)
            winner,method=shouldIStop(2,p1L, p1U, mean,epsilon=epsilon,pmargin=pmargin)
            if winner!=0:
                wilsPredicted=True
                # now to see if prediction is correct.
                if int(method) != int(2):
                    assert False
                if winner == 0:
                    assert False  # should have made a prediction.
                else:
                    if winner == best_actual:
                        # corrrect
                        storedResult = (p1.pWin, n, True, method, p1L, p1U,mean,winner,best_actual)
                        wilsonresults=storedResult
                        Wcondition2=storedResult
                    else:
                        storedResult = [p1.pWin, n, False, method, p1L, p1U, mean, winner,best_actual]
                        wilsonresults = storedResult
                        Wcondition1 = storedResult
        if not baysPredicted:
            p1L, p1U, mean = bayesian_U(p1.nWins, n, 0.05)
            winner,method=shouldIStop(1,p1L, p1U, mean,epsilon=epsilon,pmargin=pmargin)
            if winner != 0:
                baysPredicted = True
                # now to see if prediction is correct.
                if int(method) != int(1):
                    assert False
                if winner == 0:
                    assert False  # should have made a prediction.
                else:
                    if winner == best_actual:
                        # corrrect
                        storedResult = [p1.pWin, n, True, method, p1L, p1U, mean,winner,best_actual]
                        bayesresults=storedResult
                        Bcondition1=storedResult
                    else:
                        storedResult = [p1.pWin, n, False, method, p1L, p1U, mean, winner,best_actual]
                        bayesresults = storedResult
                        Bcondition1 = storedResult

            p1L, p1U, mean = bayesian_U(p1.nWins, n, 0.025)
            winner,method=shouldIStop(2,p1L, p1U, mean,epsilon=epsilon,pmargin=pmargin)
            if winner != 0:
                baysPredicted = True
                if int(method)!=int(2):
                    assert False
                # now to see if prediction is correct.
                if winner == 0:
                    assert False  # should have made a prediction.
                else:
                    if winner == best_actual:
                        # corrrect
                        storedResult = [p1.pWin, n, True, method, p1L, p1U, mean,winner,best_actual]
                        bayesresults=storedResult
                        Bcondition2=storedResult
                    else:
                        storedResult = [p1.pWin, n, False, method, p1L, p1U, mean, winner,best_actual]
                        bayesresults = storedResult
                        Bcondition2 = storedResult
        # print (f"avGames_to_predict:{avPrediction:.1f}, incorrect_Predict_rate(type 1):{(falsePredict/trials)*100:.3f}%,failed_to_predict_rate(type2) {(1-len(nGamesprediction)/trials)*100:.3f}%, predicted_n_games:{len(nGamesprediction)},  totalFailure:{(1-predictN/trials)*100:.3f}%")
    import csv
    try:
        with open(f"failureTest/wilson.csv", "a") as f:
            wr = csv.writer(f, delimiter=",")
            wr.writerow(wilsonresults)
        with open(f"failureTest/wilsonC1.csv", "a") as f:
            wr = csv.writer(f, delimiter=",")
            wr.writerow(Wcondition1)
        with open(f"failureTest/wilsonC2.csv", "a") as f:
            wr = csv.writer(f, delimiter=",")
            wr.writerow(Wcondition2)

        with open(f"failureTest/bayes.csv", "a") as f:
            wr = csv.writer(f, delimiter=",")
            wr.writerow(bayesresults)
        with open(f"failureTest/bayesC1.csv", "a") as f:
            wr = csv.writer(f, delimiter=",")
            wr.writerow(Bcondition1)
        with open(f"failureTest/bayesC2.csv", "a") as f:
            wr = csv.writer(f, delimiter=",")
            wr.writerow(Bcondition2)
    except:
        raise

    print(f"Wilson {wilsonresults}")
    print(f"Bayes{bayesresults}")

def testAccuracy(nGames,p1winrate,p2winrate=None,drawRate=None,trials=1,epsilon=0.01,pmargin=0.5):
    #np.random.seed(None)  # changed Put Outside the loop.
    #seed()
    results=dict()
    p1winrate=np.round(p1winrate,3) #without this python stores p=0.45 as 0.4499999999 which is not a draw value!!!!. unfair.

    if p2winrate==None:
        p2winrate = 1 - p1winrate  # probability of p2 winning
    if drawRate==None:
        drawRate = 1 - (p1winrate + p2winrate)

    assert (p1winrate + p2winrate) <= 1

    p1 = player(p1winrate)
    p2 = player(p2winrate)
    drawsP = player(drawRate)
    ut=0.5+epsilon #upperthreshold
    lt=0.5-epsilon #lower threshold
    if (p1.pWin > ut): #determine the actual better player
        best_actual = 1
    elif (p1.pWin < lt):
        best_actual = 2
    else:
        best_actual = 3


    wilsonresults = []
    bayesresults=[]
    Wcondition1 = []
    Wcondition2 = []
    Bcondition1 = []
    Bcondition2 = []
    for i in range(nGames):
        wilsPredicted = False
        baysPredicted = False

        g = game(p1, p2, drawsP)

        p1.reset()
        p2.reset()
        drawsP.reset()


        while not(wilsPredicted and baysPredicted): # keep going until both methods made a prediction
            #play one game
            results=playOneGame(g,results) #NB results not used
            n=p1.nWins+p2.nWins+drawsP.nWins

            if not wilsPredicted:
                p1L, p1U,mean =wils_int(p1.nWins,n,0.05)
                winner,method=shouldIStop(1,p1L, p1U, mean,epsilon=epsilon,pmargin=pmargin)
                if winner!=0:  ##condition1
                    wilsPredicted=True
                    # now to see if prediction is correct.
                    if int(method) != int(1):
                        assert False
                    if winner == 0:
                        assert False  # should have made a prediction.
                    else:
                        if winner == best_actual:
                            # corrrect
                            storedResult = [p1.pWin, n, True, method, p1L, p1U,mean,winner,best_actual,p1winrate]
                            wilsonresults.append(storedResult)
                            Wcondition1.append(storedResult)
                        else:
                            storedResult = [p1.pWin, n, False, method, p1L, p1U, mean, winner,best_actual,p1winrate]
                            wilsonresults.append(storedResult)
                            Wcondition1.append(storedResult)
                p1L, p1U, mean = wils_int(p1.nWins, n, 0.025)
                winner,method=shouldIStop(2,p1L, p1U, mean,epsilon=epsilon,pmargin=pmargin)
                if winner!=0:
                    wilsPredicted=True
                    # now to see if prediction is correct.
                    if int(method) != int(2):
                        assert False
                    if winner == 0:
                        assert False  # should have made a prediction.
                    else:
                        if winner == best_actual:
                            # corrrect
                            storedResult = (p1.pWin, n, True, method, p1L, p1U,mean,winner,best_actual,p1winrate)
                            wilsonresults.append(storedResult)
                            Wcondition2.append(storedResult)
                        else:
                            storedResult = [p1.pWin, n, False, method, p1L, p1U, mean, winner,best_actual,p1winrate]
                            wilsonresults.append(storedResult)
                            Wcondition1.append(storedResult)
            if not baysPredicted:
                p1L, p1U, mean = bayesian_U(p1.nWins, n, 0.05)
                winner,method=shouldIStop(1,p1L, p1U, mean,epsilon=epsilon,pmargin=pmargin)
                if winner != 0:
                    baysPredicted = True
                    # now to see if prediction is correct.
                    if int(method) != int(1):
                        assert False
                    if winner == 0:
                        assert False  # should have made a prediction.
                    else:
                        if winner == best_actual:
                            # corrrect
                            storedResult = [p1.pWin, n, True, method, p1L, p1U, mean,winner,best_actual,p1winrate]
                            bayesresults.append(storedResult)
                            Bcondition1.append(storedResult)
                        else:
                            storedResult = [p1.pWin, n, False, method, p1L, p1U, mean, winner,best_actual,p1winrate]
                            bayesresults.append(storedResult)
                            Bcondition1.append(storedResult)

                p1L, p1U, mean = bayesian_U(p1.nWins, n, 0.025)
                winner,method=shouldIStop(2,p1L, p1U, mean,epsilon=epsilon,pmargin=pmargin)
                if winner != 0:
                    baysPredicted = True
                    if int(method)!=int(2):
                        assert False
                    # now to see if prediction is correct.
                    if winner == 0:
                        assert False  # should have made a prediction.
                    else:
                        if winner == best_actual:
                            # corrrect
                            storedResult = [p1.pWin, n, True, method, p1L, p1U, mean,winner,best_actual,p1winrate]
                            bayesresults.append(storedResult)
                            Bcondition2.append(storedResult)
                        else:
                            storedResult = [p1.pWin, n, False, method, p1L, p1U, mean, winner,best_actual,p1winrate]
                            bayesresults.append(storedResult)
                            Bcondition2.append(storedResult)
        # print (f"avGames_to_predict:{avPrediction:.1f}, incorrect_Predict_rate(type 1):{(falsePredict/trials)*100:.3f}%,failed_to_predict_rate(type2) {(1-len(nGamesprediction)/trials)*100:.3f}%, predicted_n_games:{len(nGamesprediction)},  totalFailure:{(1-predictN/trials)*100:.3f}%")
    import csv
    try:
        with open(f"failureTest/wilson.csv", "a") as f:
            wr = csv.writer(f, delimiter=",")

            wr.writerow(wilsonresults)
        with open(f"failureTest/wilsonC1.csv", "a") as f:
            wr = csv.writer(f, delimiter=",")
            wr.writerow(Wcondition1)
        with open(f"failureTest/wilsonC2.csv", "a") as f:
            wr = csv.writer(f, delimiter=",")
            wr.writerow(Wcondition2)

        with open(f"failureTest/bayes.csv", "a") as f:
            wr = csv.writer(f, delimiter=",")
            wr.writerow(bayesresults)
        with open(f"failureTest/bayesC1.csv", "a") as f:
            wr = csv.writer(f, delimiter=",")
            wr.writerow(Bcondition1)
        with open(f"failureTest/bayesC2.csv", "a") as f:
            wr = csv.writer(f, delimiter=",")
            wr.writerow(Bcondition2)
    except:
        raise

    print(f"Wilson {wilsonresults}")
    print(f"Bayes{bayesresults}")

import random
from scipy.stats import truncnorm

def get_truncated_normal(mean=0, sd=1, low=0, upp=10):
    return truncnorm(
        (low - mean) / sd, (upp - mean) / sd, loc=mean, scale=sd)


def choosefromPoolTest(ngames=5000,drawThreshold=0.05,alpha=0.05,pmargin=0.5):
    mu, sigma = 0.5, .2  # mean and standard deviation
    trucN = get_truncated_normal(mu, sigma, 0,1)
    s=trucN.rvs(ngames)
    ##This test will select from a pool of players normally distributed around 0.5
    count, bins, ignored = plt.hist(s, 100, normed=False)
    #plt.plot(bins, 1 / (sigma * np.sqrt(2 * np.pi)) *np.exp( - (bins - mu)**2 / (2 * sigma**2) ),linewidth=2, color='r')
    plt.title("Population distribution for testing prediction")
    plt.ylabel("Quantity")
    plt.xlabel("Probability player A is better than player B")
    plt.savefig(f"failureTest/populationHist_eps:{drawThreshold}_alpha:{alpha}.png", format='png')

    plt.show()

    epsilon=drawThreshold #draw threshold
    Bcorrect=0
    Wcorrect=0
    nplayed=0
    results=dict()
    wAvGamesToPredic=[]
    bAvGamesToPredic=[]
    Plessthanpoint5minusepsilon=[]
    Plessthanpoint5andpmorethanpoint5plisepsilon=[]
    Pmorethanpoint5andplessthanpoint5plisepsilon=[]
    Pmorethanpoint5plusepsilon=[]

    wpredictiongrid={}
    wpredictiongrid['pab<0.5-epsilon']=np.zeros(3)
    wpredictiongrid['0.5-epsilon<pab and pab<0.5']=np.zeros(3)
    wpredictiongrid['0.5<pab and pab<=0.5+epsilon']=np.zeros(3)
    wpredictiongrid['pab>0.5+epsilon']=np.zeros(3)
    bpredictiongrid = {}
    bpredictiongrid['pab<0.5-epsilon'] = np.zeros(3)
    bpredictiongrid['0.5-epsilon<pab and pab<0.5'] = np.zeros(3)
    bpredictiongrid['0.5<pab and pab<=0.5+epsilon'] = np.zeros(3)
    bpredictiongrid['pab>0.5+epsilon'] = np.zeros(3)

    for p in s:
        p1 = player(p)
        p2 = player(1-p)
        drawsP=player(0)
        #g = game(p1, p2, player(0))
        ut = 0.5 + epsilon  # upperthreshold
        lt = 0.5 - epsilon  # lower threshold
        if p1.pWin>p2.pWin:
            best_actual = 1
        elif p1.pWin<p2.pWin:
            best_actual = 2
        else:
            best_actual=3
        #p=0.46193
        #p1 = player(p)
        #p2 = player(1 - p)
        p5minep = np.floor(float(0.5 - epsilon) * 1000) / 1000.0  # python rounding causes problems on the edges
        p5plusep = np.ceil(0.5 + epsilon * 1000) / 1000.0
        if p1.pWin<=p5plusep and p1.pWin>=p5minep:
            drawOK=True
        else:
            drawOK=False

        wilsPredicted = False
        baysPredicted = False
        g = game(p1, p2, player(0))
        wilsonresults = []
        bayesresults = []
        p1.reset()
        p2.reset()
        drawsP.reset()
        Wcondition1 = []
        Wcondition2 = []
        Bcondition1 = []
        Bcondition2 = []

        while not (wilsPredicted and baysPredicted):  # keep going until both methods made a prediction
            # play one game
            results = playOneGame(g, results)  # NB results not used
            n = p1.nWins + p2.nWins + drawsP.nWins

            if not wilsPredicted:
                #########################WILSON CONDITION 1
                p1L, p1U, mean = wils_int(p1.nWins, n, alpha)
                winner, method = shouldIStop(1, p1L, p1U, mean, epsilon=0,pmargin=pmargin) #no threshold for lcb only
                if winner != 0:  #condition1
                    wilsPredicted = True
                    # now to see if prediction is correct.
                    if int(method) != int(1):
                        assert False #should only do predict 1
                    if winner == 0:
                        assert False  # should have made a prediction.
                    else:
                        if winner == best_actual:
                            # corrrect
                            storedResult = [p1.pWin, p1.nWins,n, True, method, p1L, p1U, mean, winner, best_actual,p1U-p1L]
                            wilsonresults = storedResult
                            Wcondition1 = storedResult
                            if drawOK: #it could've been a draw.
                                storedResult = [p1.pWin,p1.nWins, n, True, method, p1L, p1U, mean, winner, best_actual,
                                                p1U - p1L,"or Draw"]
                                wilsonresults = storedResult
                                Wcondition2 = storedResult

                        elif winner==3 and drawOK: #it's within the draw threshold
                            storedResult = [p1.pWin,p1.nWins, n, True, method, p1L, p1U, mean, winner, best_actual,p1U-p1L]
                            wilsonresults = storedResult
                            Wcondition1 = storedResult
                        else: #it is wrong
                            storedResult = [p1.pWin,p1.nWins, n, False, method, p1L, p1U, mean, winner, best_actual,p1U-p1L]
                            wilsonresults = storedResult
                            Wcondition1 = storedResult
                #########################WILSON CONDITION 2
                #p1.nWins=798
                #n=1573
                p1L, p1U, mean = wils_int(p1.nWins, n, alpha/2)
                winner, method = shouldIStop(2, p1L, p1U, mean, epsilon=epsilon,pmargin=pmargin)
                if winner != 0:
                    wilsPredicted = True
                    # now to see if prediction is correct.
                    if int(method) != int(2):
                        assert False
                    if winner == 0:
                        assert False  # should have made a prediction.
                    else:
                        if winner == best_actual:
                            # corrrect
                            storedResult = (p1.pWin,p1.nWins, n, True, method, p1L, p1U, mean, winner, best_actual,p1U-p1L)
                            wilsonresults = storedResult
                            Wcondition2 = storedResult
                            if drawOK: #it could've been a draw.
                                storedResult = [p1.pWin,p1.nWins, n, True, method, p1L, p1U, mean, winner, best_actual,
                                                p1U - p1L,"or Draw"]
                                wilsonresults = storedResult
                                Wcondition2 = storedResult

                        elif winner==3 and drawOK: #it's within the draw threshold
                            storedResult = (p1.pWin,p1.nWins, n, True, method, p1L, p1U, mean, winner, best_actual, p1U - p1L)
                            wilsonresults = storedResult
                            Wcondition2 = storedResult
                        else:
                            storedResult = [p1.pWin,p1.nWins, n, False, method, p1L, p1U, mean, winner, best_actual,p1U-p1L]
                            wilsonresults = storedResult
                            Wcondition1 = storedResult



            if not baysPredicted:
                #########################BAYES CONDITION 1

                p1L, p1U, mean = bayesian_U(p1.nWins, n, alpha)
                winner, method = shouldIStop(1, p1L, p1U, mean, epsilon=0,pmargin=pmargin)
                if winner != 0:
                    baysPredicted = True
                    # now to see if prediction is correct.
                    if int(method) != int(1):
                        assert False
                    if winner == 0:
                        assert False  # should have made a prediction.
                    else:
                        if winner == best_actual:
                            # corrrect
                            storedResult = [p1.pWin,p1.nWins, n, True, method, p1L, p1U, mean, winner, best_actual,p1U-p1L]

                            if drawOK: #it could've been a draw.
                                storedResult = [p1.pWin,p1.nWins, n, True, method, p1L, p1U, mean, winner, best_actual,
                                                p1U - p1L,"or Draw"]
                            bayesresults = storedResult
                            Bcondition1 = storedResult
                        elif winner==3 and drawOK: #it's within the draw threshold
                            storedResult = (p1.pWin,p1.nWins, n, True, method, p1L, p1U, mean, winner, best_actual, p1U - p1L)
                            bayesresults = storedResult
                            Bcondition1 = storedResult
                        else:
                            storedResult = [p1.pWin,p1.nWins, n, False, method, p1L, p1U, mean, winner, best_actual,p1U-p1L]
                            bayesresults = storedResult
                            Bcondition1 = storedResult
                #########################BAYES CONDITION 2
                p1L, p1U, mean = bayesian_U(p1.nWins, n, alpha/2)
                winner, method = shouldIStop(2, p1L, p1U, mean, epsilon=epsilon,pmargin=pmargin)
                if winner != 0:
                    baysPredicted = True
                    if int(method) != int(2):
                        assert False
                    # now to see if prediction is correct.
                    if winner == 0:
                        assert False  # should have made a prediction.

                    else:
                        if winner == best_actual:
                            # corrrect
                            storedResult = [p1.pWin,p1.nWins, n, True, method, p1L, p1U, mean, winner, best_actual,p1U-p1L]
                            if drawOK: #it could've been a draw.
                                storedResult = [p1.pWin,p1.nWins, n, True, method, p1L, p1U, mean, winner, best_actual,
                                                p1U - p1L,"or Draw"]

                            bayesresults = storedResult
                            Bcondition2 = storedResult
                        elif winner == 3 and drawOK:  # it's within the draw threshold
                            storedResult = (p1.pWin,p1.nWins, n, True, method, p1L, p1U, mean, winner, best_actual, p1U - p1L)
                            bayesresults = storedResult
                            Bcondition1 = storedResult
                        else:
                            storedResult = [p1.pWin,p1.nWins, n, False, method, p1L, p1U, mean, winner, best_actual,p1U-p1L]
                            bayesresults = storedResult
                            Bcondition2 = storedResult
            # print (f"avGames_to_predict:{avPrediction:.1f}, incorrect_Predict_rate(type 1):{(falsePredict/trials)*100:.3f}%,failed_to_predict_rate(type2) {(1-len(nGamesprediction)/trials)*100:.3f}%, predicted_n_games:{len(nGamesprediction)},  totalFailure:{(1-predictN/trials)*100:.3f}%")

        if bayesresults[3]:
            Bcorrect+=1
        if wilsonresults[3]:
            Wcorrect+=1
        nplayed+=1
        wAvGamesToPredic.append(wilsonresults[2])
        bAvGamesToPredic.append(bayesresults[2])

        import csv
        try:
            with open(f"failureTest/wilson.csv", "a") as f:
                wr = csv.writer(f, delimiter=",")
                wr.writerow(wilsonresults)
            with open(f"failureTest/wilsonC1.csv", "a") as f:
                wr = csv.writer(f, delimiter=",")
                wr.writerow(Wcondition1)
            with open(f"failureTest/wilsonC2.csv", "a") as f:
                wr = csv.writer(f, delimiter=",")
                wr.writerow(Wcondition2)

            with open(f"failureTest/bayes.csv", "a") as f:
                wr = csv.writer(f, delimiter=",")
                wr.writerow(bayesresults)
            with open(f"failureTest/bayesC1.csv", "a") as f:
                wr = csv.writer(f, delimiter=",")
                wr.writerow(Bcondition1)
            with open(f"failureTest/bayesC2.csv", "a") as f:
                wr = csv.writer(f, delimiter=",")
                wr.writerow(Bcondition2)
        except:
            raise

        #print(f"Wilson {wilsonresults}")
        #print(f"Bayes{bayesresults}")
        #####NOW save data for selectiong table

        #Plessthanpoint5minusepsilon = []
        #Pmorethanpoint5minusepsilon = []
        ##Pmorethanpoint5morethanpoint5minusepison = []
        #Plessthanpoint5plusepsilonmorethanpoint5 = []
        wthisbin=None
        bthisbin=None

        pab=p1.pWin  #TODO: Implement this in the other test
        p5minep=np.floor(float(0.5-epsilon)*1000)/1000.0 #python rounding causes problems on the edges
        p5plusep=np.ceil(float(0.5+epsilon)*1000)/1000.0
        if pab<p5minep:
            wthisbin=wpredictiongrid['pab<0.5-epsilon']
            bthisbin=bpredictiongrid['pab<0.5-epsilon']

        elif p5minep<pab and pab<0.5:
            wthisbin=wpredictiongrid['0.5-epsilon<pab and pab<0.5']
            bthisbin=bpredictiongrid['0.5-epsilon<pab and pab<0.5']

        elif 0.5<pab and pab<=p5plusep:
            wthisbin=wpredictiongrid['0.5<pab and pab<=0.5+epsilon']
            bthisbin=bpredictiongrid['0.5<pab and pab<=0.5+epsilon']

        elif pab>p5plusep:
            wthisbin=wpredictiongrid['pab>0.5+epsilon']
            bthisbin=bpredictiongrid['pab>0.5+epsilon']
        else:
            assert False

        wprediction=wilsonresults[8]
        if wprediction==1:
            wthisbin[2]+=1
        elif wprediction==2:
            wthisbin[0]+=1
        elif wprediction==3:
            wthisbin[1]+=1
        bprediction = bayesresults[8]
        if bprediction == 1:
            bthisbin[2] += 1
        elif bprediction == 2:
            bthisbin[0] += 1
        elif bprediction == 3:
            bthisbin[1] += 1
        else:
            assert False

        if nplayed%100==0:
            print("**********************************************************************************************")
            print(f"Type\tAv Games\tnCorrect\tnplayed\t\taccuracy")
            print(f"Wils\t{np.round(np.mean(wAvGamesToPredic),2)}\t\t{Wcorrect}\t\t{nplayed}\t\t{Wcorrect/nplayed}")
            print(f"bayes\t{np.round(np.mean(bAvGamesToPredic),2)}\t\t{Bcorrect}\t\t{nplayed}\t\t{Bcorrect/nplayed}")

            print("________________________________________________________________________________________")
            print(f"wilson___epsilon={epsilon}________________________________________")
            width = 30
            lw=6
            line='{0: <{width}}'.format("actual   \predicted ->", width=width)
            line += "|{0:<7}|{1:<7}|{2:<7}".format("B", "Draw", "A")
            print(line)

            for key,v in wpredictiongrid.items():
                line='{0: <{width}}'.format(key, width=width)
                line+="|{:<7}|{:<7}|{:<7}".format(int(v[0]),int(v[1]),int(v[2]))
                print(line)
            print("________________________________________________________________________________________")
            #print(wpredictiongrid)
            print(f"Bayes_______epsilon={epsilon}_____________________________________")
            line='{0: <{width}}'.format("actual   \predicted ->", width=width)
            line += "|{0:<7}|{1:<7}|{2:<7}".format("B", "Draw", "A")
            print(line)
            for key, v in bpredictiongrid.items():
                line='{0: <{width}}'.format(key, width=width)
                line+="|{:<7}|{:<7}|{:<7}".format(int(v[0]),int(v[1]),int(v[2]))
                print(line)

            #print(bpredictiongrid)
            print("**********************************************************************************************")

    import csv

    with open(f"failureTest/accuracyTest_eps{epsilon}_alpha_{alpha}_predicMargin={pmargin}.txt", "w") as f:

        line=f"Type\tngames\tAv Games\tnCorrect\t% "
        f.write(line+"\n")
        print(line)
        line=f"Wils\t{ngames}\t{np.round(np.mean(wAvGamesToPredic),2)}\t\t{Wcorrect}\t\t{Wcorrect/nplayed}"
        f.write(line+"\n")
        print(line)
        line=f"bayes\t{ngames}\t{np.round(np.mean(bAvGamesToPredic),2)}\t\t{Bcorrect}\t\t{Bcorrect/nplayed}"
        f.write(line+"\n")
        print(line)
        line=f"wilson_____epsilon={epsilon}______________________________________"
        f.write(line+"\n")
        width = 30
        lw = 6
        line = '{0: <{width}}'.format("actual   \predicted ->", width=width)
        line += "|{0:<7}|{1:<7}|{2:<7}".format("B", "Draw", "A")
        f.write(line+"\n")
        print(line)
        for key, v in wpredictiongrid.items():
            line = '{0: <{width}}'.format(key, width=width)
            line += "|{:<7}|{:<7}|{:<7}".format(int(v[0]), int(v[1]), int(v[2]))
            f.write(line+"\n")
            print(line)
        # print(wpredictiongrid)
        line=f"Bayes_______epsilon={epsilon}_____________________________________"
        f.write(line+"\n")
        print(line)
        line = '{0: <{width}}'.format("actual   \predicted ->", width=width)
        line += "|{0:<7}|{1:<7}|{2:<7}".format("B", "Draw", "A")
        f.write(line+"\n")
        print(line)
        for key, v in bpredictiongrid.items():
            line = '{0: <{width}}'.format(key, width=width)
            line += "|{:<7}|{:<7}|{:<7}".format(int(v[0]), int(v[1]), int(v[2]))
            f.writelines(line+"\n")
            print(line)

def coverageTest(ngames=5000,drawThreshold=0.05,alpha=0.05,pmargin=0.5):
    #Iterates over a series of p values and records the coverage for that value.


    epsilon=drawThreshold #draw threshold
    Bcorrect=0
    Wcorrect=0
    nplayed=0
    results=dict()
    wAvGamesToPredic=[]
    bAvGamesToPredic=[]

    wilX=[]
    wilY=[]
    bayX=[]
    bayY=[]
    wpredictiongrid={}
    wpredictiongrid['pab<0.5-epsilon']=np.zeros(3)
    wpredictiongrid['0.5-epsilon<pab and pab<0.5']=np.zeros(3)
    wpredictiongrid['0.5<pab and pab<=0.5+epsilon']=np.zeros(3)
    wpredictiongrid['pab>0.5+epsilon']=np.zeros(3)
    bpredictiongrid = {}
    bpredictiongrid['pab<0.5-epsilonepsilon'] = np.zeros(3)
    bpredictiongrid['0.5-epsilon<pab and pab<0.5'] = np.zeros(3)
    bpredictiongrid['0.5<pab and pab<=0.5+epsilon'] = np.zeros(3)
    bpredictiongrid['pab>0.5+epsilon'] = np.zeros(3)
    s=np.arange(0.11,0.9,0.027)
    #s=[0.5]
    for p in s:
        p=np.round(p,3) #without this python stores p=0.45 as 0.4499999999 which is not a draw value!!!!. unfair.
        wcorrect = []
        bcorrect = []
        p1 = player(p)
        p2 = player(1-p)
        drawsP=player(0)
        #g = game(p1, p2, player(0))
        #ut = 0.5 + epsilon  # upperthreshold
        #lt = 0.5 - epsilon  # lower threshold
        if p1.pWin>p2.pWin:
            best_actual = 1
        elif p1.pWin<p2.pWin:
            best_actual = 2
        else:
            best_actual=3
        #p=0.46193
        #p1 = player(p)
        #p2 = player(1 - p)
        if p1.pWin<=0.5+epsilon and p1.pWin>=0.5-epsilon:
            drawOK=True
        else:
            drawOK=False
        for i in range(ngames):
            ################THIS IS WHERE THE LOOP WILL GO TO GET COVERAGE FOR A SPECIFIC VALUE
            wilsPredicted = False
            baysPredicted = False
            g = game(p1, p2, player(0))
            wilsonresults = []
            bayesresults = []
            p1.reset()
            p2.reset()
            drawsP.reset()
            Wcondition1 = []
            Wcondition2 = []
            Bcondition1 = []
            Bcondition2 = []

            while not (wilsPredicted and baysPredicted):  # keep going until both methods made a prediction
                # play one game
                results = playOneGame(g, results)  # NB results not used
                n = p1.nWins + p2.nWins + drawsP.nWins

                if not wilsPredicted:
                    #########################WILSON CONDITION 1
                    #p1.pWin = 0.57
                    #p1.nWins = 823
                    #n = 1507
                    p1L, p1U, mean = wils_int(p1.nWins, n, alpha)
                    #p1L=np.round(p1L,3)
                    #p1U = np.round(p1U  , 3)
                    #mean = np.round(mean, 3)

                    winner, method = shouldIStop(1, p1L, p1U, mean, epsilon=epsilon,pmargin=pmargin) #no threshold for lcb only
                    if winner != 0:  #condition1
                        wilsPredicted = True
                        # now to see if prediction is correct.
                        if int(method) != int(1):
                            assert False #should only do predict 1
                        if winner == 0:
                            assert False  # should have made a prediction.
                        else:
                            if winner == best_actual:
                                # corrrect
                                storedResult = [p1.pWin, p1.nWins,n, True, method, p1L, p1U, mean, winner, best_actual,p1U-p1L]
                                wilsonresults = storedResult
                                Wcondition1 = storedResult
                                if drawOK: #it could've been a draw.
                                    storedResult = [p1.pWin,p1.nWins, n, True, method, p1L, p1U, mean, winner, best_actual,
                                                    p1U - p1L,"or Draw"]
                                    wilsonresults = storedResult
                                    Wcondition2 = storedResult

                            elif winner==3 and drawOK: #it's within the draw threshold
                                storedResult = [p1.pWin,p1.nWins, n, True, method, p1L, p1U, mean, winner, best_actual,p1U-p1L]
                                wilsonresults = storedResult
                                Wcondition1 = storedResult
                            else: #it is wrong
                                storedResult = [p1.pWin,p1.nWins, n, False, method, p1L, p1U, mean, winner, best_actual,p1U-p1L]
                                wilsonresults = storedResult
                                Wcondition1 = storedResult
                                print(f"wils{method} failed. {storedResult}")

                    #########################WILSON CONDITION 2
                    #p1.nWins=798
                    #n=1573
                    #p1.pWin=0.57
                    #p1.nWins=823
                    #n=1507

                    p1L, p1U, mean = wils_int(p1.nWins, n, alpha/2)
                    #p1L = np.round(p1L, 3)
                    #p1U = np.round(p1U, 3)
                    #mean = np.round(mean, 3)

                    winner, method = shouldIStop(2, p1L, p1U, mean, epsilon=epsilon,pmargin=pmargin)
                    if winner != 0 and not wilsPredicted:
                        wilsPredicted = True
                        # now to see if prediction is correct.
                        if int(method) != int(2):
                            assert False
                        if winner == 0:
                            assert False  # should have made a prediction.
                        else:
                            if winner == best_actual:
                                # corrrect
                                storedResult = (p1.pWin,p1.nWins, n, True, method, p1L, p1U, mean, winner, best_actual,p1U-p1L)
                                wilsonresults = storedResult
                                Wcondition2 = storedResult
                                if drawOK: #it could've been a draw.
                                    storedResult = [p1.pWin,p1.nWins, n, True, method, p1L, p1U, mean, winner, best_actual,
                                                    p1U - p1L,"or Draw"]
                                    wilsonresults = storedResult
                                    Wcondition2 = storedResult

                            elif winner==3 and drawOK: #it's within the draw threshold
                                storedResult = (p1.pWin,p1.nWins, n, True, method, p1L, p1U, mean, winner, best_actual, p1U - p1L)
                                wilsonresults = storedResult
                                Wcondition2 = storedResult
                            else:
                                storedResult = [p1.pWin,p1.nWins, n, False, method, p1L, p1U, mean, winner, best_actual,p1U-p1L]
                                wilsonresults = storedResult
                                Wcondition1 = storedResult
                                print(f"wils{method} failed. {storedResult}")


                if not baysPredicted:
                    #########################BAYES CONDITION 1

                    p1L, p1U, mean = bayesian_U(p1.nWins, n, alpha)
                    p1L = np.round(p1L, 3)
                    p1U = np.round(p1U, 3)
                    mean = np.round(mean, 3)
                    winner, method = shouldIStop(1, p1L, p1U, mean, epsilon=epsilon,pmargin=pmargin)
                    if winner != 0:
                        baysPredicted = True
                        # now to see if prediction is correct.
                        if int(method) != int(1):
                            assert False
                        if winner == 0:
                            assert False  # should have made a prediction.
                        else:
                            if winner == best_actual:
                                # corrrect
                                storedResult = [p1.pWin,p1.nWins, n, True, method, p1L, p1U, mean, winner, best_actual,p1U-p1L]

                                if drawOK: #it could've been a draw.
                                    storedResult = [p1.pWin,p1.nWins, n, True, method, p1L, p1U, mean, winner, best_actual,
                                                    p1U - p1L,"or Draw"]
                                bayesresults = storedResult
                                Bcondition1 = storedResult
                            elif winner==3 and drawOK: #it's within the draw threshold
                                storedResult = (p1.pWin,p1.nWins, n, True, method, p1L, p1U, mean, winner, best_actual, p1U - p1L)
                                bayesresults = storedResult
                                Bcondition1 = storedResult
                            else:
                                storedResult = [p1.pWin,p1.nWins, n, False, method, p1L, p1U, mean, winner, best_actual,p1U-p1L]
                                bayesresults = storedResult
                                Bcondition1 = storedResult
                                print(f"bayes{method} failed. {storedResult}")

                    #########################BAYES CONDITION 2
                    p1L, p1U, mean = bayesian_U(p1.nWins, n, alpha/2)
                    #p1L = np.round(p1L, 3)
                    #p1U = np.round(p1U, 3)
                    #mean = np.round(mean, 3)
                    winner, method = shouldIStop(2, p1L, p1U, mean, epsilon=epsilon,pmargin=pmargin)
                    if winner != 0 and not baysPredicted:
                        baysPredicted = True
                        if int(method) != int(2):
                            assert False
                        # now to see if prediction is correct.
                        if winner == 0:
                            assert False  # should have made a prediction.

                        else:
                            if winner == best_actual:
                                # corrrect
                                storedResult = [p1.pWin,p1.nWins, n, True, method, p1L, p1U, mean, winner, best_actual,p1U-p1L]
                                if drawOK: #it could've been a draw.
                                    storedResult = [p1.pWin,p1.nWins, n, True, method, p1L, p1U, mean, winner, best_actual,
                                                    p1U - p1L,"or Draw"]

                                bayesresults = storedResult
                                Bcondition2 = storedResult
                            elif winner == 3 and drawOK:  # it's within the draw threshold
                                storedResult = (p1.pWin,p1.nWins, n, True, method, p1L, p1U, mean, winner, best_actual, p1U - p1L)
                                bayesresults = storedResult
                                Bcondition1 = storedResult
                            else:
                                storedResult = [p1.pWin,p1.nWins, n, False, method, p1L, p1U, mean, winner, best_actual,p1U-p1L]
                                print(f"bayes{method} failed. {storedResult}")
                                bayesresults = storedResult
                                Bcondition2 = storedResult

            if bayesresults[3]:
                Bcorrect+=1
                bcorrect.append(1)
            else:
                bcorrect.append(0)

            if wilsonresults[3]:
                Wcorrect+=1
                wcorrect.append(1)
            else:
                wcorrect.append(0)

            nplayed+=1
            wAvGamesToPredic.append(wilsonresults[2])
            bAvGamesToPredic.append(bayesresults[2])

        ######################LOOP ENDS HERE
        import csv

        try:
            with open(f"failureTest/wcoverage.csv", "a") as f:
                nCorrect = np.count_nonzero(wcorrect)
                n = len(wcorrect)
                wilsonC = np.average(wcorrect)
                predictLength=np.average(wAvGamesToPredic)
                wilX.append(p1.pWin)
                wilY.append(wilsonC)
                line=[nCorrect,n,wilsonC,predictLength]
                wr = csv.writer(f, delimiter=",")
                wr.writerow(line)

            with open(f"failureTest/bcoverage.csv", "a") as f:
                nCorrect = np.count_nonzero(bcorrect)
                n = len(bcorrect)
                bayesC = np.average(bcorrect)
                predictLength = np.average(bAvGamesToPredic)
                bayX.append(p1.pWin)
                bayY.append(bayesC)
                line=[nCorrect,n,bayesC,predictLength]
                wr = csv.writer(f, delimiter=",")
                wr.writerow(line)


        except:
            raise

        print(f"________________Wilson  eps={epsilon}_______________________________")
        np.set_printoptions(precision=5)
        print(f"{np.round(wilX,5)}")
        print(f"{np.round(wilY,5)}")
        print(f"________________Bayes  eps={epsilon}________________________________")
        print(f"{np.round(bayX,5)}")
        print(f"{np.round(bayY,5)}")

    fig1 = plt.figure(figsize=plt.figaspect(0.5))
    ax = fig1.add_subplot(1, 1, 1)

    ax.plot(wilX,wilY)
    ax.set_title(f"Coverage using Wilson.alpha={alpha}  epsilon={epsilon} predictmargin={pmargin}")
    fig1.savefig(f"failureTest/wilsoncoverage_alpha={alpha}_epsilon={epsilon}_predicMargin={pmargin}.png",format="png")
    fig2 = plt.figure(figsize=plt.figaspect(0.5))
    ax2 = fig2.add_subplot(1, 1, 1)
    ax2.set_title(f"Coverage using Bayesian-U. alpha={alpha} epsilon={epsilon} predictmargin={pmargin}")

    ax2.plot(bayX, bayY)
    fig2.savefig(f"failureTest/bayescoverage_alpha={alpha}_epsilon={epsilon}_predicMargin={pmargin}.png",format="png")
    plt.show()

if __name__ == '__main__':
    np.random.seed(None)  # changed Put Outside the loop.
    random.seed()

    fullResult=dict()
    alpha=0.05
    coverageTest(ngames=5000,drawThreshold=0.05,alpha=alpha,pmargin=0.5)
    choosefromPoolTest(ngames=5000,drawThreshold=0.05,alpha=alpha,pmargin=0.5)



    pass
