
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
def playGames(p1winrate,p2winrate=None,drawRate=None,trials=1,epsilon=0.01):

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
            winner,method=shouldIStop(1,p1L, p1U, mean,epsilon=epsilon)
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
            winner,method=shouldIStop(2,p1L, p1U, mean,epsilon=epsilon)
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
            winner,method=shouldIStop(1,p1L, p1U, mean,epsilon=epsilon)
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
            winner,method=shouldIStop(2,p1L, p1U, mean,epsilon=epsilon)
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

def testAccuracy(nGames,p1winrate,p2winrate=None,drawRate=None,trials=1,epsilon=0.01):
    #np.random.seed(None)  # changed Put Outside the loop.
    #seed()
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
                winner,method=shouldIStop(1,p1L, p1U, mean,epsilon=epsilon)
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
                winner,method=shouldIStop(2,p1L, p1U, mean,epsilon=epsilon)
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
                winner,method=shouldIStop(1,p1L, p1U, mean,epsilon=epsilon)
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
                winner,method=shouldIStop(2,p1L, p1U, mean,epsilon=epsilon)
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


def choosefromPoolTest(ngames=5000,drawThreshold=0.05,alpha=0.05):
    mu, sigma = 0.5, .2  # mean and standard deviation
    trucN = get_truncated_normal(mu, sigma, 0,1)
    s=trucN.rvs(ngames)
    ##This test will select from a pool of players normally distributed around 0.5
    count, bins, ignored = plt.hist(s, 100, normed=False)
    #plt.plot(bins, 1 / (sigma * np.sqrt(2 * np.pi)) *np.exp( - (bins - mu)**2 / (2 * sigma**2) ),linewidth=2, color='r')
    plt.title("Population distribution for testing prediction")
    plt.ylabel("Quantity")
    plt.xlabel("Probability player A is better than player B")
    plt.savefig("populationHist.png", format='png')

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
        if p1.pWin<=0.5+epsilon and p1.pWin>=0.5-epsilon:
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
                winner, method = shouldIStop(1, p1L, p1U, mean, epsilon=0) #no threshold for lcb only
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
                winner, method = shouldIStop(2, p1L, p1U, mean, epsilon=epsilon)
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
                winner, method = shouldIStop(1, p1L, p1U, mean, epsilon=0)
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
                winner, method = shouldIStop(2, p1L, p1U, mean, epsilon=epsilon)
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

        pab=p1.pWin
        if pab<0.5-epsilon:
            wthisbin=wpredictiongrid['pab<0.5-epsilon']
            bthisbin=bpredictiongrid['pab<0.5-epsilon']

        elif 0.5-epsilon<pab and pab<0.5:
            wthisbin=wpredictiongrid['0.5-epsilon<pab and pab<0.5']
            bthisbin=bpredictiongrid['0.5-epsilon<pab and pab<0.5']

        elif 0.5<pab and pab<=0.5+epsilon:
            wthisbin=wpredictiongrid['0.5<pab and pab<=0.5+epsilon']
            bthisbin=bpredictiongrid['0.5<pab and pab<=0.5+epsilon']

        elif pab>0.5+epsilon:
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
            print("****************************************************************************")
            print(f"Type\tAv Games\tnCorrect\tnplayed\t\taccuracy")
            print(f"Wils\t{np.round(np.mean(wAvGamesToPredic),2)}\t\t{Wcorrect}\t\t{nplayed}\t\t{Wcorrect/nplayed}")
            print(f"bayes\t{np.round(np.mean(bAvGamesToPredic),2)}\t\t{Bcorrect}\t\t{nplayed}\t\t{Bcorrect/nplayed}")

            print("wilson___________________________________________")
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
            print("Bayes____________________________________________")
            line='{0: <{width}}'.format("actual   \predicted ->", width=width)
            line += "|{0:<7}|{1:<7}|{2:<7}".format("B", "Draw", "A")
            print(line)
            for key, v in bpredictiongrid.items():
                line='{0: <{width}}'.format(key, width=width)
                line+="|{:<7}|{:<7}|{:<7}".format(int(v[0]),int(v[1]),int(v[2]))
                print(line)

            #print(bpredictiongrid)
            print("****************************************************************************")



    print(f"Type\t{ngames}\tAv Games\tnCorrect\t%")
    print(f"Wils\t{ngames}\t{np.round(np.mean(wAvGamesToPredic),2)}\t\t{Wcorrect}\t\t{Wcorrect/nplayed}")
    print(f"bayes\t{ngames}\t{np.round(np.mean(bAvGamesToPredic),2)}\t\t{Bcorrect}\t\t{Bcorrect/nplayed}")
    print("wilson___________________________________________")
    width = 30
    lw = 6
    line = '{0: <{width}}'.format("actual   \predicted ->", width=width)
    line += "|{0:<7}|{1:<7}|{2:<7}".format("B", "Draw", "A")
    print(line)

    for key, v in wpredictiongrid.items():
        line = '{0: <{width}}'.format(key, width=width)
        line += "|{:<7}|{:<7}|{:<7}".format(int(v[0]), int(v[1]), int(v[2]))
        print(line)
    # print(wpredictiongrid)
    print("Bayes____________________________________________")
    line = '{0: <{width}}'.format("actual   \predicted ->", width=width)
    line += "|{0:<7}|{1:<7}|{2:<7}".format("B", "Draw", "A")
    print(line)
    for key, v in bpredictiongrid.items():
        line = '{0: <{width}}'.format(key, width=width)
        line += "|{:<7}|{:<7}|{:<7}".format(int(v[0]), int(v[1]), int(v[2]))
        print(line)
if __name__ == '__main__':
    np.random.seed(None)  # changed Put Outside the loop.
    random.seed()

    fullResult=dict()

    choosefromPoolTest(ngames=10000)
    assert False
    for pab in np.arange(0.45,.55,0.001) :

        playGames(p1winrate=pab,epsilon=0.05)


    pass
