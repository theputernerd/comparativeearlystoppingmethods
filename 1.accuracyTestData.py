"""
This creates a table of accuracy for a range of different parameters.
"""
import matplotlib as mpl
mpl.use('Agg')
import numpy as np
import random
from lib import player
playerPoolDist = random.uniform(1.5, 1.9)
from wilson import wils_int
from lib import shouldIStop,game,ConfusionMatrix,drawOk
from bayes import bayesian_U
tests = [bayesian_U, wils_int]
import time

import matplotlib.pyplot as plt


def playOneGame(g, results):
    winner = g.playGame()
    # winner = drawsP #change remove me.
    winner.nWins += 1
    return results


def accuracyTest(nGames, p1winrate, p2winrate=None, drawRate=None, trials=1, epsilon=0.01, delta=0.5):
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
                stop,winner= shouldIStop(1, p1L, p1U, mean, n, delta=delta, epsilon=epsilon)
                method = 1
                if stop:  ##condition1
                    wilsPredicted=True
                    # now to see if prediction is correct.
                    method=1
                    if int(method) == int(3):
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
                stop, winner= shouldIStop(3, p1L, p1U, mean, n, delta=delta, epsilon=epsilon)
                method = 3
                if stop and not wilsPredicted:
                    wilsPredicted=True
                    # now to see if prediction is correct.
                    method=3
                    if int(method) != int(3):
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
                stop, winner= shouldIStop(1, p1L, p1U, mean, n, delta=delta, epsilon=epsilon)
                method = 1
                if stop:
                    baysPredicted = True
                    # now to see if prediction is correct.
                    method=1
                    if int(method) == int(3):
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
                stop, winner= shouldIStop(3, p1L, p1U, mean, n, delta=delta, epsilon=epsilon)
                method = 3
                if stop and not baysPredicted:
                    baysPredicted = True
                    method=3
                    if int(method)!=int(3):
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
from scipy.stats import truncnorm,uniform
import scipy
def get_truncated_normal(mean=0, sd=1, low=0, upp=10):
    return uniform()
    return truncnorm(
        (low - mean) / sd, (upp - mean) / sd, loc=mean, scale=sd)

def choosefromPoolTest(population,ngames=5000, epsilon=0.05, alpha=0.05, delta=0.5,prefix=""):

    #s=population.rvs(ngames)
    s=population#np.arange(0.0,1.0,1.0/ngames)
    fig3 = plt.figure(figsize=plt.figaspect(0.5))
    ax3 = fig3.add_subplot(1, 1, 1)
    ##This test will select from a pool of players normally distributed around 0.5
    count, bins, ignored = plt.hist(s, 100, normed=False)
    #plt.plot(bins, 1 / (sigma * np.sqrt(2 * np.pi)) *np.exp( - (bins - mu)**2 / (2 * sigma**2) ),linewidth=2, color='r')
    ax3.set_title("Population distribution for testing prediction")
    ax3.set_ylabel("Quantity")
    ax3.set_xlabel("Probability player A is better than player B")
    print(f"writing failureTest/{prefix}populationHist_eps{epsilon}_alpha{alpha}.png")
    fig3.savefig(f"failureTest/{prefix}populationHist_eps{epsilon}_alpha{alpha}.png", format='png')

    #plt.show()

    epsilon=epsilon #draw threshold
    Bcorrect=0
    Wcorrect=0
    nplayed=0
    results=dict()
    wAvGamesToPredic=[]
    bAvGamesToPredic=[]

    wpredictiongrid = {}
    wpredictiongrid['Pab<0.5-delta'] = np.zeros(3)
    wpredictiongrid['0.5-delta<=Pab<0.5'] = np.zeros(3)
    wpredictiongrid['0.5<Pab<=0.5+delta'] = np.zeros(3)
    wpredictiongrid['Pab>0.5+delta'] = np.zeros(3)
    wpredictiongrid['Pab==0.5'] = np.zeros(3)
    bpredictiongrid = {}
    bpredictiongrid['Pab<0.5-delta'] = np.zeros(3)
    bpredictiongrid['0.5-delta<=Pab<0.5'] = np.zeros(3)
    bpredictiongrid['0.5<Pab<=0.5+delta'] = np.zeros(3)
    bpredictiongrid['Pab>0.5+delta'] = np.zeros(3)
    bpredictiongrid['Pab==0.5'] = np.zeros(3)

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
        drawOK=drawOk(p1.pWin,delta)


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
            #p1.nWins = 2871
            #p2.nWins = 6364 - 2871
            #p1.pWin = 0.434
            #p2.pWin = 1 - 0.434
            #best_actual=2
            n = p1.nWins + p2.nWins + drawsP.nWins

            if not wilsPredicted:
                #########################WILSON CONDITION 1
                p1L, p1U, mean = wils_int(p1.nWins, n, alpha)
                stop, winner = shouldIStop(1, p1L, p1U, mean, n, delta=delta, epsilon=0)  #no threshold for lcb only
                method = 1
                if stop:
                    wilsPredicted = True
                    # now to see if prediction is correct.
                    method=1
                    if int(method) == int(3):
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
                                                p1U - p1L,"or Similar"]
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
                stop, winner = shouldIStop(3, p1L, p1U, mean, n, delta=delta, epsilon=epsilon)
                method = 3
                if stop and not wilsPredicted:
                    wilsPredicted = True
                    # now to see if prediction is correct.
                    method=3
                    if int(method) != int(3):
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
                                                p1U - p1L,"or Similar"]
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
                stop, winner = shouldIStop(1, p1L, p1U, mean, n, delta=delta, epsilon=0)
                method = 1
                if stop:
                    baysPredicted = True
                    # now to see if prediction is correct.
                    method=1
                    if int(method) == int(3):
                        assert False
                    if winner == 0:
                        assert False  # should have made a prediction.
                    else:
                        if winner == best_actual:
                            # corrrect
                            storedResult = [p1.pWin,p1.nWins, n, True, method, p1L, p1U, mean, winner, best_actual,p1U-p1L]

                            if drawOK: #it could've been a draw.
                                storedResult = [p1.pWin,p1.nWins, n, True, method, p1L, p1U, mean, winner, best_actual,
                                                p1U - p1L,"or Similar"]
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
                stop, winner = shouldIStop(3, p1L, p1U, mean, n, delta=delta, epsilon=epsilon)
                method = 3
                if stop and not baysPredicted:
                    baysPredicted = True
                    method=3
                    if int(method) != int(3):
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
                                                p1U - p1L,"or Similar"]

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
            with open(f"failureTest/{prefix}wilson.csv", "a",newline="") as f:
                wr = csv.writer(f, delimiter=",")
                wr.writerow(wilsonresults)
            with open(f"failureTest/{prefix}wilsonC1.csv", "a",newline="") as f:
                wr = csv.writer(f, delimiter=",")
                wr.writerow(Wcondition1)
            with open(f"failureTest/{prefix}wilsonC2.csv", "a",newline="") as f:
                wr = csv.writer(f, delimiter=",")
                wr.writerow(Wcondition2)

            with open(f"failureTest/{prefix}bayes.csv", "a",newline="") as f:
                wr = csv.writer(f, delimiter=",")
                wr.writerow(bayesresults)
            with open(f"failureTest/{prefix}bayesC1.csv", "a",newline="") as f:
                wr = csv.writer(f, delimiter=",")
                wr.writerow(Bcondition1)
            with open(f"failureTest/{prefix}bayesC2.csv", "a",newline="") as f:
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
        binNum=None
        if pab<0.5-delta:
            wthisbin=wpredictiongrid['Pab<0.5-delta']
            bthisbin=bpredictiongrid['Pab<0.5-delta']
            binNum=1
        elif 0.5-delta<=pab and pab<0.5:
            wthisbin=wpredictiongrid['0.5-delta<=Pab<0.5']
            bthisbin=bpredictiongrid['0.5-delta<=Pab<0.5']
            binNum=2

        elif 0.5<pab and pab<=0.5+delta:
            wthisbin=wpredictiongrid['0.5<Pab<=0.5+delta']
            bthisbin=bpredictiongrid['0.5<Pab<=0.5+delta']
            binNum=3

        elif pab==0.5:
            wthisbin=wpredictiongrid['Pab==0.5']
            bthisbin=bpredictiongrid['Pab==0.5']
            binNum=4


        elif pab>0.5+delta:
            wthisbin=wpredictiongrid['Pab>0.5+delta']
            bthisbin=bpredictiongrid['Pab>0.5+delta']
            binNum=5

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
            bins=[wpredictiongrid, bpredictiongrid]
            for bin in bins:
                combinedDrawsA=bin['0.5<Pab<=0.5+delta'][0]+bin['0.5-delta<=Pab<0.5'][0]+bin['Pab==0.5'][0]
                combinedDrawsB = bin['0.5<Pab<=0.5+delta'][1] + bin['0.5-delta<=Pab<0.5'][1] + \
                                 bin['Pab==0.5'][1]
                combinedDrawsC = bin['0.5<Pab<=0.5+delta'][2] + bin['0.5-delta<=Pab<0.5'][2] + \
                                 bin['Pab==0.5'][2]
                bin['Similar']=[combinedDrawsA,combinedDrawsB,combinedDrawsC]

            print("**********************************************************************************************")
            print(f"Type\tAv Games\tnCorrect\tnplayed\t\taccuracy")
            print(f"Wils\t{np.round(np.mean(wAvGamesToPredic),2)}\t\t{Wcorrect}\t\t{nplayed}\t\t{Wcorrect/nplayed}")
            print(f"bayes\t{np.round(np.mean(bAvGamesToPredic),2)}\t\t{Bcorrect}\t\t{nplayed}\t\t{Bcorrect/nplayed}")

            print("________________________________________________________________________________________")
            print(f"wilson___alpha:{alpha}________________________________________")
            width = 30
            lw=6
            line='{0: <{width}}'.format("actual   \predicted ->", width=width)
            line += "|{0:<7}|{1:<7}|{2:<7}|{3:<7}".format("B", "Similar", "A", "Ngames")
            print(line)
            keys=[]
            vs=[]
            [(keys.append(key),vs.append(v)) for key,v in wpredictiongrid.items()]
            its=sorted(zip(keys,vs))
            for key,v in its:
                line='{0: <{width}}'.format(key, width=width)
                line += "|{:<7}|{:<7}|{:<7}|{:<7}".format(np.round(v[0] / np.sum(v), 2), np.round(v[1] / np.sum(v), 2),
                                                          np.round(v[2] / np.sum(v), 2), int(np.sum(v)))
                print(line)
            print("________________________________________________________________________________________")
            #print(wpredictiongrid)
            print(f"Bayes_______epislon={epsilon}_____________________________________")
            line='{0: <{width}}'.format("actual   \predicted ->", width=width)
            line += "|{0:<7}|{1:<7}|{2:<7}|{3:<7}".format("B", "Similar", "A", "Ngames")
            print(line)
            for key, v in bpredictiongrid.items():
                line='{0: <{width}}'.format(key, width=width)
                line += "|{:<7}|{:<7}|{:<7}|{:<7}".format(np.round(v[0] / np.sum(v), 2), np.round(v[1] / np.sum(v), 2),
                                                          np.round(v[2] / np.sum(v), 2), int(np.sum(v)))
                print(line)

            #print(bpredictiongrid)
            print("**********************************************************************************************")

    import csv
    bins = [wpredictiongrid, bpredictiongrid]
    for bin in bins:
        combinedDrawsA = bin['0.5<Pab<=0.5+delta'][0] + bin['0.5-delta<=Pab<0.5'][0] + bin['Pab==0.5'][
            0]
        combinedDrawsB = bin['0.5<Pab<=0.5+delta'][1] + bin['0.5-delta<=Pab<0.5'][1] + \
                         bin['Pab==0.5'][1]
        combinedDrawsC = bin['0.5<Pab<=0.5+delta'][2] + bin['0.5-delta<=Pab<0.5'][2] + \
                         bin['Pab==0.5'][2]
        bin['Similar'] = [combinedDrawsA, combinedDrawsB, combinedDrawsC]

    with open(f"failureTest/{prefix}accuracyTest_eps{epsilon}_alpha_{alpha}_predicMargin={delta}.txt", "w") as f:

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
        line += "|{0:<7}|{1:<7}|{2:<7}|{3:<7}".format("B", "Similar", "A", "Ngames")
        f.write(line+"\n")
        print(line)
        for key, v in wpredictiongrid.items():
            line = '{0: <{width}}'.format(key, width=width)
            if np.sum(v)!=0:

                try:
                    line += "|{:<7}|{:<7}|{:<7}|{:<7}".format(np.round(v[0] / np.sum(v),2), np.round(v[1] / np.sum(v),2), np.round(v[2] / np.sum(v),2),int(np.sum(v)))
                except:
                    pass
            else:
                line += "|{:<7}|{:<7}|{:<7}|{:<7}".format(0,0,0,0)

            f.write(line+"\n")
            print(line)
        # print(wpredictiongrid)
        line=f"Bayes_______alpha={alpha} ngames={np.sum(v)}_____________________________________"
        f.write(line+"\n")
        print(line)
        line = '{0: <{width}}'.format("actual   \predicted ->", width=width)
        line += "|{0:<7}|{1:<7}|{2:<7}|{3:<7}".format("B", "Similar", "A", "Ngames")
        f.write(line+"\n")
        print(line)
        for key, v in bpredictiongrid.items():
            line = '{0: <{width}}'.format(key, width=width)
            if np.sum(v)!=0:

                line += "|{:<7}|{:<7}|{:<7}|{:<7}".format(np.round(v[0] / np.sum(v),2), np.round(v[1] / np.sum(v),2), np.round(v[2] / np.sum(v),2),int(np.sum(v)))
            else:
                line += "|{:<7}|{:<7}|{:<7}|{:<7}".format(0,0,0,0)
            f.writelines(line+"\n")
            print(line)
def C1ConfusionMatrix_fromPoolTest(population,ngames=5000, epsilon=0.05, alpha=0.05, delta=0.5,prefix=""):
    s=population #np.arange(0.0,1.0,1.0/ngames)

    #s=population.rvs(ngames)
    fig3 = plt.figure(figsize=plt.figaspect(0.5))
    ax3 = fig3.add_subplot(1, 1, 1)
    ##This test will select from a pool of players normally distributed around 0.5
    count, bins, ignored = plt.hist(s, 100, normed=False)
    #plt.plot(bins, 1 / (sigma * np.sqrt(2 * np.pi)) *np.exp( - (bins - mu)**2 / (2 * sigma**2) ),linewidth=2, color='r')
    ax3.set_title(f"{prefix} Population distribution for testing prediction")
    ax3.set_ylabel("Quantity")
    ax3.set_xlabel("Probability player A is better than player B")
    print(f"writing confusionMatrix/{prefix}populationHist_alpha{alpha}.png")
    fig3.savefig(f"confusionMatrix/{prefix}populationHist_alpha{alpha}_delta{delta}.png", format='png')

    #plt.show()

    nplayed=0
    results=dict()
    wAvGamesToPredic=[]
    bAvGamesToPredic=[]

    Wc1ConfMatrix = ConfusionMatrix(f"Wils C1",predictions=["A>B","B>A"])
    Wc2ConfMatrix = ConfusionMatrix(f"Wils C2",predictions=["Similar","NOT(Similar)"])
    Wcombined = ConfusionMatrix(f"Wils Combined",predictions=["",""])
    Bc1ConfMatrix = ConfusionMatrix(f"Bayes C1", predictions=["A>B","B>A"])
    Bc2ConfMatrix = ConfusionMatrix(f"Bayes C2", predictions=["Similar", "NOT(Similar)"])
    Bcombined = ConfusionMatrix(f"Bayes Combined", predictions=["", ""])
    CMs=[Bc1ConfMatrix,Bc2ConfMatrix,Bcombined,Wc1ConfMatrix,Wc2ConfMatrix,Wcombined]

    for p in s:
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

        drawOK=drawOk(p1.pWin,delta)


        wilsPredicted = False
        baysPredicted = False

        g = game(p1, p2, player(0))

        p1.reset()
        p2.reset()
        drawsP.reset()

        while not (wilsPredicted and baysPredicted):  # keep going until both methods made a prediction
            # play one game
            results = playOneGame(g, results)  # NB results not used
            n = p1.nWins + p2.nWins + drawsP.nWins
            if not wilsPredicted:
                #########################WILSON CONDITION 1
                p1L, p1U, mean = wils_int(p1.nWins, n, alpha)
                stop, winner = shouldIStop(1, p1L, p1U, mean, n, delta=delta, epsilon=0)  #no threshold for lcb only
                method = 1
                if stop:
                    wilsPredicted = True
                    # now to see if prediction is correct.
                    if winner ==1:
                        if best_actual==1:
                            Wc1ConfMatrix.TP+=1
                            Wcombined.TP+=1
                        else:
                            Wc1ConfMatrix.FP+=1
                            Wcombined.FP+=1

                    elif winner==2:
                        if best_actual==2:
                            Wcombined.TN+=1
                            Wc1ConfMatrix.TN+=1

                        else:
                            Wcombined.FN+=1
                            Wc1ConfMatrix.FN+=1

                    elif winner==3:
                        assert False #didnt expect to get a draw from c1

                    else:
                        assert False #unknown winner
                    wilsNgames=n
                    Wcombined.gamesToPredict.append(n)
                    Wc1ConfMatrix.gamesToPredict.append(n)

                #########################WILSON CONDITION 2

                if not wilsPredicted:
                    p1L, p1U, mean = wils_int(p1.nWins, n, alpha/2.0)
                    stop, winner = shouldIStop(3, p1L, p1U, mean, n, delta=delta, epsilon=0)

                method = 3
                if stop and not wilsPredicted:
                    wilsPredicted=True
                    if winner!=3:
                        assert False #expecting prediction fo draw only.

                    elif winner == 3:
                        if drawOK:
                            Wcombined.TP+=1
                            Wc2ConfMatrix.TP+=1
                        elif not drawOK:
                            Wcombined.FP+=1

                            Wc2ConfMatrix.FP+=1
                    else:
                        assert False  # unknown winner
                    Wcombined.gamesToPredict.append(n)
                    Wc2ConfMatrix.gamesToPredict.append(n)
                    wilsNgames=n
            if not baysPredicted:
                #########################BAYES CONDITION 1
                p1L, p1U, mean = bayesian_U(p1.nWins, n, alpha)
                stop, winner = shouldIStop(1, p1L, p1U, mean, n, delta=delta, epsilon=0)
                method = 1
                if stop:
                    bayesNgames=n
                    baysPredicted = True
                    # now to see if prediction is correct.
                    Bcombined.gamesToPredict.append(n)
                    Bc1ConfMatrix.gamesToPredict.append(n)
                    if winner == 1:
                        if best_actual == 1:
                            Bc1ConfMatrix.TP+=1
                            Bcombined.TP+=1

                        else:
                            Bc1ConfMatrix.FP+=1
                            Bcombined.FP+=1

                    elif winner == 2:
                        if best_actual == 2:
                            Bcombined.TN+=1
                            Bc1ConfMatrix.TN+=1

                        else:
                            Bcombined.FN+=1
                            Bc1ConfMatrix.FN+=1

                    elif winner == 3:
                        assert False  # didnt expect to get a draw from c1

                    else:
                        assert False  # unknown winner

                #########################BAYES CONDITION 2
                if not baysPredicted:
                    p1L, p1U, mean = bayesian_U(p1.nWins, n, alpha/2)
                    method = 3
                    stop, winner = shouldIStop(3, p1L, p1U, mean, n, delta=delta, epsilon=epsilon)
                if stop and not baysPredicted:
                    bayesNgames=n
                    baysPredicted = True
                    if winner!=3:
                        assert False #expecting prediction fo draw only.

                    elif winner == 3:
                        Bcombined.gamesToPredict.append(n)
                        Bc2ConfMatrix.gamesToPredict.append(n)
                        if drawOK:
                            Bcombined.TP+=1
                            Bc2ConfMatrix.TP+=1
                        elif not drawOK:
                            Bcombined.FP+=1
                            Bc2ConfMatrix.FP+=1
                    else:
                        assert False  # unknown winner


        wAvGamesToPredic.append(wilsNgames)
        bAvGamesToPredic.append(bayesNgames)

        nplayed+=1
        if nplayed%100==0:

            print("**********************************************************************************************")
            for cm in CMs:
                st=str(cm.name)+f"_AvGames:{cm.av_predict}"
                print(st)
                print(cm)

                print("----------------------------------------------------------------------------------------------")

            print("**********************************************************************************************")

    import csv
        #allmetrics

    #f=_eps{epsilon}_alpha_{alpha}_delta={delta}

    for cm in CMs:
        with open(f"confusionMatrix/{prefix}{cm.name}.csv", 'a', newline='') as f:
            csvwriter=csv.writer(f, delimiter=',', quoting=csv.QUOTE_MINIMAL)
            params=[alpha,delta]+cm.allmetrics
            csvwriter.writerow(params)

    with open(f"confusionMatrix/{prefix}_alpha_{alpha}_delta={delta}.txt", "w") as f:
        for cm in CMs:
            f.write(f"{cm.name} avgames:{cm.av_predict}\n")
            f.writelines(str(cm))
            f.write("----------------------------------------------------------------------------------------------\n")
    #CMs=[Bc1ConfMatrix,Bc2ConfMatrix,Bcombined,Wc1ConfMatrix,Wc2ConfMatrix,Wcombined]

    return [Wcombined.ACC,Wcombined.av_predict,Bcombined.ACC,Bcombined.av_predict]

def getACCfor3d_fixedPAB(ngames,alpha,delta,pab):
        p = pab
        p1 = player(p)
        p2 = player(1 - p)
        drawsP = player(0)

        nplayed = 0
        wAvGamesToPredic = []
        bAvGamesToPredic = []

        Wc1ConfMatrix = ConfusionMatrix(f"Wils C1", predictions=["A>B", "B>A"])
        Wc2ConfMatrix = ConfusionMatrix(f"Wils C2", predictions=["Similar", "NOT(Similar)"])
        Wcombined = ConfusionMatrix(f"Wils Combined", predictions=["", ""])
        Bc1ConfMatrix = ConfusionMatrix(f"Bayes C1", predictions=["A>B", "B>A"])
        Bc2ConfMatrix = ConfusionMatrix(f"Bayes C2", predictions=["Similar", "NOT(Similar)"])
        Bcombined = ConfusionMatrix(f"Bayes Combined", predictions=["", ""])
        CMs = [Bc1ConfMatrix, Bc2ConfMatrix, Bcombined, Wc1ConfMatrix, Wc2ConfMatrix, Wcombined]

        if p1.pWin > p2.pWin:
            best_actual = 1
        elif p1.pWin < p2.pWin:
            best_actual = 2
        else:
            best_actual = 3

        if abs(p1.pWin - p2.pWin) <= delta:
            drawOK = True
        else:
            drawOK = False
        aW=[]
        aB=[]
        for i in range(ngames):
            wilsPredicted = False
            baysPredicted = False
            results = dict()

            g = game(p1, p2, player(0))

            p1.reset()
            p2.reset()
            drawsP.reset()

            while not (wilsPredicted and baysPredicted):  # keep going until both methods made a prediction
                # play one game
                results = playOneGame(g, results)  # NB results not used
                n = p1.nWins + p2.nWins + drawsP.nWins
                if not wilsPredicted:
                    #########################WILSON CONDITION 1
                    p1L, p1U, mean = wils_int(p1.nWins, n, alpha)
                    stop, winner = shouldIStop(1, p1L, p1U, mean, n, delta=delta, epsilon=0)  # no threshold for lcb only
                    method=1
                    if stop:
                        wilsPredicted = True
                        # now to see if prediction is correct.
                        if winner == 1:
                            if best_actual == 1:
                                Wc1ConfMatrix.TP += 1
                                Wcombined.TP += 1
                            else:
                                Wc1ConfMatrix.FP += 1
                                Wcombined.FP += 1

                        elif winner == 2:
                            if best_actual == 2:
                                Wcombined.TN += 1
                                Wc1ConfMatrix.TN += 1

                            else:
                                Wcombined.FN += 1
                                Wc1ConfMatrix.FN += 1

                        elif winner == 3:
                            assert False  # didnt expect to get a draw from c1

                        else:
                            assert False  # unknown winner
                        wilsNgames = n
                        Wcombined.gamesToPredict.append(n)
                        Wc1ConfMatrix.gamesToPredict.append(n)

                    #########################WILSON CONDITION 2

                    if not wilsPredicted:
                        p1L, p1U, mean = wils_int(p1.nWins, n, alpha / 2.0)
                        stop, winner = shouldIStop(3, p1L, p1U, mean, n, delta=delta, epsilon=0)
                    method=3

                    if stop and not wilsPredicted:
                        method = 3
                        wilsPredicted = True
                        if winner != 3:
                            assert False  # expecting prediction fo draw only.

                        elif winner == 3:
                            if drawOK:
                                Wcombined.TP += 1
                                Wc2ConfMatrix.TP += 1
                            elif not drawOK:
                                Wcombined.FP += 1

                                Wc2ConfMatrix.FP += 1
                        else:
                            assert False  # unknown winner
                        Wcombined.gamesToPredict.append(n)
                        Wc2ConfMatrix.gamesToPredict.append(n)
                        wilsNgames = n
                if not baysPredicted:
                    #########################BAYES CONDITION 1
                    p1L, p1U, mean = bayesian_U(p1.nWins, n, alpha)
                    stop, winner = shouldIStop(1, p1L, p1U, mean, n, delta=delta, epsilon=0)
                    method = 1
                    if stop:
                        method=1
                        bayesNgames = n
                        baysPredicted = True
                        # now to see if prediction is correct.
                        Bcombined.gamesToPredict.append(n)
                        Bc1ConfMatrix.gamesToPredict.append(n)
                        if winner == 1:
                            if best_actual == 1:
                                Bc1ConfMatrix.TP += 1
                                Bcombined.TP += 1

                            else:
                                Bc1ConfMatrix.FP += 1
                                Bcombined.FP += 1

                        elif winner == 2:
                            if best_actual == 2:
                                Bcombined.TN += 1
                                Bc1ConfMatrix.TN += 1

                            else:
                                Bcombined.FN += 1
                                Bc1ConfMatrix.FN += 1

                        elif winner == 3:
                            assert False  # didnt expect to get a draw from c1

                        else:
                            assert False  # unknown winner

                    #########################BAYES CONDITION 2
                    if not baysPredicted:
                        p1L, p1U, mean = bayesian_U(p1.nWins, n, alpha / 2)
                        stop, winner = shouldIStop(3, p1L, p1U, mean, n, delta=delta, epsilon=0)
                    method = 3
                    if stop and not baysPredicted:
                        method=3
                        bayesNgames = n
                        baysPredicted = True
                        if winner != 3:
                            assert False  # expecting prediction fo draw only.

                        elif winner == 3:
                            Bcombined.gamesToPredict.append(n)
                            Bc2ConfMatrix.gamesToPredict.append(n)
                            if drawOK:
                                Bcombined.TP += 1
                                Bc2ConfMatrix.TP += 1
                            elif not drawOK:
                                Bcombined.FP += 1
                                Bc2ConfMatrix.FP += 1
                        else:
                            assert False  # unknown winner

            wAvGamesToPredic.append(wilsNgames)
            bAvGamesToPredic.append(bayesNgames)

            aW.append(Wcombined.ACC) #accuracy
            aB.append(Bcombined.ACC)  # accuracy

        ngW=Wcombined.av_predict #.average(wAvGamesToPredic)
        ngB = Bcombined.av_predict #.average(bAvGamesToPredic)
        Waccuracyrate = Wcombined.ACC
        Baccuracyrate = Bcombined.ACC
        return [[Waccuracyrate,ngW],[Baccuracyrate,ngB]]


def coverageTest(ngames=5000, epsilon=0.00, alpha=0.05, delta=0.5):
    #Iterates over a series of p values and records the coverage for that value.

    epsilon=epsilon #draw threshold
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

    #s=np.arange(0.20,0.8,0.007)
    s=np.arange(0.1,0.91,0.01)
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
        predictWhenDrawn = True  # set to tru to allow a prediciton when it is a draw
        if predictWhenDrawn:
            if p1.pWin>p2.pWin:
                best_actual = 1
            elif p1.pWin<p2.pWin:
                best_actual = 2
            else:
                best_actual=3
        else:
            assert False #is this really what you want?
            if p1.pWin>0.5+delta:
                best_actual = 1
            elif p1.pWin<0.5-delta:
                best_actual = 2
            else:
                best_actual=3

        drawOK=drawOk(p1.pWin,delta)

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
                    p1L=np.round(p1L,3)
                    p1U = np.round(p1U  , 3)
                    mean = np.round(mean, 3)
                    method=1
                    stop, winner = shouldIStop(1, p1L, p1U, mean, n, delta=delta,
                                               epsilon=epsilon)  #no threshold for lcb only
                    if stop:  #condition1
                        wilsPredicted = True
                        # now to see if prediction is correct.

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
                                                    p1U - p1L,"or Similar"]
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
                    p1L = np.round(p1L, 3)
                    p1U = np.round(p1U, 3)
                    mean = np.round(mean, 3)

                    stop, winner = shouldIStop(3, p1L, p1U, mean, n, delta=delta, epsilon=epsilon)
                    method = 3
                    if stop and not wilsPredicted:
                        wilsPredicted = True
                        # now to see if prediction is correct.
                        method=3
                        if int(method) != int(3):
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
                                                    p1U - p1L,"or Similar"]
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

                    stop, winner= shouldIStop(1, p1L, p1U, mean, n, delta=delta, epsilon=epsilon)
                    method = 1
                    if stop :
                        baysPredicted = True
                        # now to see if prediction is correct.
                        method=1
                        if int(method) == int(3):
                            assert False
                        if winner == 0:
                            assert False  # should have made a prediction.
                        else:
                            if winner == best_actual:
                                # corrrect
                                storedResult = [p1.pWin,p1.nWins, n, True, method, p1L, p1U, mean, winner, best_actual,p1U-p1L]

                                if drawOK: #it could've been a draw.
                                    storedResult = [p1.pWin,p1.nWins, n, True, method, p1L, p1U, mean, winner, best_actual,
                                                    p1U - p1L,"or Similar"]
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
                    p1L = np.round(p1L, 3)
                    p1U = np.round(p1U, 3)
                    mean = np.round(mean, 3)
                    stop, winner = shouldIStop(3, p1L, p1U, mean, n, delta=delta, epsilon=epsilon)
                    method = 3
                    if stop and not baysPredicted:
                        baysPredicted = True
                        method=3
                        if int(method) != int(3):
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
                                                    p1U - p1L,"or Similar"]

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
            print("Writing to 1.accuracyforParams/wcoverage.csv")
            with open(f"1.accuracyforParams/wcoverage.csv", "a") as f:
                nCorrect = np.count_nonzero(wcorrect)
                n = len(wcorrect)
                wilsonC = np.average(wcorrect)
                predictLength=np.average(wAvGamesToPredic)
                wilX.append(p1.pWin)
                wilY.append(wilsonC)
                line=[nCorrect,n,wilsonC,predictLength]
                wr = csv.writer(f, delimiter=",")
                wr.writerow(line)
            print("Writing to 1.accuracyforParams/bcoverage.csv")
            with open(f"1.accuracyforParams/bcoverage.csv", "a") as f:
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

    ax.plot(wilX,wilY,'x')
    ax.set_title(f"Accuracy using Wilson.alpha={alpha} delta={delta}")
    ax.set_xlabel(f"True probability Pab")
    ax.set_ylabel(f"Prediction accuracy")
    print(f"Writing 1.accuracyforParams/wilsoncoverage_alpha={alpha}_epsilon={epsilon}_delta={delta}.png")
    fig1.savefig(f"1.accuracyforParams/wilsoncoverage_alpha={alpha}_epsilon={epsilon}_delta={delta}.png",format="png")
    fig2 = plt.figure(figsize=plt.figaspect(0.5))
    ax2 = fig2.add_subplot(1, 1, 1)
    ax2.set_title(f"Accuracy using Bayesian-U. alpha={alpha} delta={delta}")
    ax2.set_xlabel(f"True probability Pab")
    ax2.set_ylabel(f"Prediction accuracy")
    ax2.plot(bayX, bayY,'x')
    print(f"Writing 1.accuracyforParams/bayescoverage_alpha={alpha}_epsilon={epsilon}_delta={delta}.png")
    fig2.savefig(f"1.accuracyforParams/bayescoverage_alpha={alpha}_epsilon={epsilon}_delta={delta}.png",format="png")
    #plt.show()

def getaccuracy3dfixedpab(ngames,alpha,delta,pab):
    #returns xyz values for accuracy. y is accuracy, x,y is alpha delta
    Bcorrect = 0
    Wcorrect = 0
    nplayed = 0
    wAvGamesToPredic=[]
    bAvGamesToPredic=[]
    wcorrect = []
    bcorrect = []
    p=pab
    p1 = player(p)
    p2 = player(1 - p)
    drawsP = player(0)
    epsilon=0
    if p1.pWin > p2.pWin:
        best_actual = 1
    elif p1.pWin < p2.pWin:
        best_actual = 2
    else:
        best_actual = 3
    drawOK = drawOk(p1.pWin, delta)

    for i in range(ngames):
        wilsPredicted = False
        baysPredicted = False
        g = game(p1, p2, player(0))
        wilsonresults = []
        bayesresults = []
        results = dict()
        p1.reset()
        p2.reset()
        drawsP.reset()
        while not (wilsPredicted and baysPredicted):  # keep going until both methods made a prediction
            # play one game
            results = playOneGame(g, results)  # NB results not used
            n = p1.nWins + p2.nWins + drawsP.nWins

            if not wilsPredicted: # and n > 7: #todo: remove this.
                #########################WILSON CONDITION 1
                # p1.pWin = 0.57
                # p1.nWins = 823
                # n = 1507
                p1L, p1U, mean = wils_int(p1.nWins, n, alpha)
                p1L = np.round(p1L, 3)
                p1U = np.round(p1U, 3)
                mean = np.round(mean, 3)

                stop, winner = shouldIStop(1, p1L, p1U, mean, n, delta=delta, epsilon=epsilon)  # no threshold for lcb only
                method = 1
                if stop :  # condition1
                    wilsPredicted = True
                    # now to see if prediction is correct.
                    method=1
                    if int(method) == int(3):
                        assert False  # should only do predict 1
                    if winner == 0:
                        assert False  # should have made a prediction.
                    else:
                        if winner == best_actual:
                            # corrrect
                            storedResult = [p1.pWin, p1.nWins, n, True, method, p1L, p1U, mean, winner, best_actual,
                                            p1U - p1L]
                            wilsonresults = storedResult
                            Wcondition1 = storedResult
                            if drawOK:  # it could've been a draw.
                                storedResult = [p1.pWin, p1.nWins, n, True, method, p1L, p1U, mean, winner, best_actual,
                                                p1U - p1L, "or Similar"]
                                wilsonresults = storedResult
                                Wcondition2 = storedResult

                        elif winner == 3 and drawOK:  # it's within the draw threshold
                            storedResult = [p1.pWin, p1.nWins, n, True, method, p1L, p1U, mean, winner, best_actual,
                                            p1U - p1L]
                            wilsonresults = storedResult
                            Wcondition1 = storedResult
                        else:  # it is wrong
                            storedResult = [p1.pWin, p1.nWins, n, False, method, p1L, p1U, mean, winner, best_actual,
                                            p1U - p1L]
                            wilsonresults = storedResult
                            Wcondition1 = storedResult
                            print(f"wils{method} failed. {storedResult}")

                #########################WILSON CONDITION 2
                # p1.nWins=798
                # n=1573
                # p1.pWin=0.57
                # p1.nWins=823
                # n=1507

                p1L, p1U, mean = wils_int(p1.nWins, n, alpha / 2)
                p1L = np.round(p1L, 3)
                p1U = np.round(p1U, 3)
                mean = np.round(mean, 3)

                stop, winner = shouldIStop(3, p1L, p1U, mean, n, delta=delta, epsilon=epsilon)
                method = 3
                if stop  and not wilsPredicted:
                    wilsPredicted = True
                    # now to see if prediction is correct.
                    method=3
                    if int(method) != int(3):
                        assert False
                    if winner == 0:
                        assert False  # should have made a prediction.
                    else:
                        if winner == best_actual:
                            # corrrect
                            storedResult = (
                            p1.pWin, p1.nWins, n, True, method, p1L, p1U, mean, winner, best_actual, p1U - p1L)
                            wilsonresults = storedResult
                            Wcondition2 = storedResult
                            if drawOK:  # it could've been a draw.
                                storedResult = [p1.pWin, p1.nWins, n, True, method, p1L, p1U, mean, winner, best_actual,
                                                p1U - p1L, "or Similar"]
                                wilsonresults = storedResult
                                Wcondition2 = storedResult

                        elif winner == 3 and drawOK:  # it's within the draw threshold
                            storedResult = (
                            p1.pWin, p1.nWins, n, True, method, p1L, p1U, mean, winner, best_actual, p1U - p1L)
                            wilsonresults = storedResult
                            Wcondition2 = storedResult
                        else:
                            storedResult = [p1.pWin, p1.nWins, n, False, method, p1L, p1U, mean, winner, best_actual,
                                            p1U - p1L]
                            wilsonresults = storedResult
                            Wcondition1 = storedResult
                            print(f"wils{method} failed. {storedResult}")

            if not baysPredicted:
                #########################BAYES CONDITION 1

                p1L, p1U, mean = bayesian_U(p1.nWins, n, alpha)

                stop, winner = shouldIStop(1, p1L, p1U, mean, n, delta=delta, epsilon=epsilon)
                method = 1
                if stop:
                    stop = False
                    method=1
                    baysPredicted = True
                    # now to see if prediction is correct.
                    if int(method) == int(3):
                        assert False
                    if winner == 0:
                        assert False  # should have made a prediction.
                    else:
                        if winner == best_actual:
                            # corrrect
                            storedResult = [p1.pWin, p1.nWins, n, True, method, p1L, p1U, mean, winner, best_actual,
                                            p1U - p1L]

                            if drawOK:  # it could've been a draw.
                                storedResult = [p1.pWin, p1.nWins, n, True, method, p1L, p1U, mean, winner, best_actual,
                                                p1U - p1L, "or Similar"]
                            bayesresults = storedResult
                            Bcondition1 = storedResult
                        elif winner == 3 and drawOK:  # it's within the draw threshold
                            storedResult = (
                            p1.pWin, p1.nWins, n, True, method, p1L, p1U, mean, winner, best_actual, p1U - p1L)
                            bayesresults = storedResult
                            Bcondition1 = storedResult
                        else:
                            storedResult = [p1.pWin, p1.nWins, n, False, method, p1L, p1U, mean, winner, best_actual,
                                            p1U - p1L]
                            bayesresults = storedResult
                            Bcondition1 = storedResult
                            print(f"bayes{method} failed. {storedResult}")

                #########################BAYES CONDITION 2
                p1L, p1U, mean = bayesian_U(p1.nWins, n, alpha / 2.0)
                p1L = np.round(p1L, 3)
                p1U = np.round(p1U, 3)
                mean = np.round(mean, 3)
                stop, winner = shouldIStop(3, p1L, p1U, mean, n, delta=delta, epsilon=epsilon)
                method = 3
                if stop and not baysPredicted:
                    method==3
                    baysPredicted = True
                    if int(method) != int(3):
                        assert False
                    # now to see if prediction is correct.
                    if winner == 0:
                        assert False  # should have made a prediction.

                    else:
                        if winner == best_actual:
                            # corrrect
                            storedResult = [p1.pWin, p1.nWins, n, True, method, p1L, p1U, mean, winner, best_actual,
                                            p1U - p1L]
                            if drawOK:  # it could've been a draw.
                                storedResult = [p1.pWin, p1.nWins, n, True, method, p1L, p1U, mean, winner, best_actual,
                                                p1U - p1L, "or Similar"]

                            bayesresults = storedResult
                            Bcondition2 = storedResult
                        elif winner == 3 and drawOK:  # it's within the draw threshold
                            storedResult = (
                            p1.pWin, p1.nWins, n, True, method, p1L, p1U, mean, winner, best_actual, p1U - p1L)
                            bayesresults = storedResult
                            Bcondition1 = storedResult
                        else:
                            storedResult = [p1.pWin, p1.nWins, n, False, method, p1L, p1U, mean, winner, best_actual,
                                            p1U - p1L]
                            print(f"bayes{method} failed. {storedResult}")
                            bayesresults = storedResult
                            Bcondition2 = storedResult

        if bayesresults[3]:
            Bcorrect += 1
            bcorrect.append(1)
        else:
            bcorrect.append(0)

        if wilsonresults[3]:
            Wcorrect += 1
            wcorrect.append(1)
        else:
            wcorrect.append(0)

        nplayed += 1
        wAvGamesToPredic.append(wilsonresults[2])
        bAvGamesToPredic.append(bayesresults[2])
    nWCorrect = np.count_nonzero(wcorrect)
    nW = len(wcorrect)
    aW = np.average(wcorrect) #accuracy
    ngW=np.average(wAvGamesToPredic) #how many game to predict
    Wdel=delta
    Walph=alpha
    Waccuracyrate=aW

    nBCorrect = np.count_nonzero(bcorrect)
    nB = len(bcorrect)
    aB = np.average(bcorrect)  # accuracy
    ngB = np.average(bAvGamesToPredic)  # how many game to predict
    Bdel = delta
    Balph = alpha
    Baccuracyrate = aB

    return [[Waccuracyrate,ngW],[Baccuracyrate,ngB]]
import numpy
def interpretXYZ(x,y,z,pts=1000):
    from scipy.interpolate import griddata
    import random
    if len(x) > 0:
        x, y,z = zip(*random.sample(list(zip(x, y,z)), int(len(x) / 1)))
    xi = np.linspace(min(x), max(x),pts)
    yi = np.linspace(min(y), max(y),pts)
    zi = griddata((x, y), z, (xi[None,:], yi[:,None]), method='cubic')
    return xi,yi,zi


import multiprocessing
if __name__ == '__main__':

    np.random.seed(None)  # changed Put Outside the loop.
    random.seed()
    import os
    try:
        os.mkdir('confusionMatrix')
    except:
        print(f"confusionMatrix Folder exists")
    try:
        os.mkdir("1.accuracyforParams")
    except FileExistsError:
        pass

    try:
        os.mkdir("failureTest")
    except FileExistsError:
        pass
    #assert False
    fullResult=dict()
    #alpha=0.05
    #alpha=0.0423
    epsilon=0.0
    delta=0.05
    alphaList=[0.05,0.01,0.1,0.02]
    deltaList=[.1,0.2,0.05,0.04]
    #######plt.title(r'$\alpha > \beta$')
    mu, sigma = 0.5, 0  # mean and standard deviation
    ngames=5000
    population = np.arange(0.0,1.0,1.0/ngames)#get_truncated_normal(mu, sigma, 0,1)
    #population=np.random.uniform(0.1,0.9,100)
    poolsize=10
    jobs=[]

    for a in alphaList:

        #[C1ConfusionMatrix_fromPoolTest(population, ngames=5000, epsilon=epsilon, alpha=a, delta=d) for d in deltaList]

        for d in deltaList:
            print(f"(alpha,delta) ({a},{d})")
            print("Getting Data for PoolTest")
            # choosefromPoolTest(population, ngames=1000, epsilon=epsilon, alpha=a, delta=d)
            # population,ngames=5000, epsilon=0.05, alpha=0.05, delta=0.5
            # writes to failureTest
            p = multiprocessing.Process(target=choosefromPoolTest, args=(population, ngames, epsilon, a, d))
            jobs.append(p)
            p.start()
            #This does the same job but for a fixed p
            print(f"(alpha,delta) ({a},{d})")
            print("Getting Data for PoolTest Fixed p=0.5")
            prob = np.full(ngames / 3.0, 0.5)  # (0.5,0.5,1.0/ngames)
            plessthanpointfive = np.arange(0.3, 0.65, ngames * 2.0 / 3.0)
            p = multiprocessing.Process(target=choosefromPoolTest, args=(prob, ngames, epsilon, a, d,f"fixedp{0.5}"))
            jobs.append(p)
            p.start()
            while len(jobs) >= poolsize:  # this is my pool
                # check if any are closed.
                for j in jobs:
                    if not j.is_alive():
                        print(f"-removing {p.pid}")
                        jobs.remove(j)
                time.sleep(1)
            print(f"(alpha,delta) ({a},{d})")
            print("Getting Data for CoverageTest")
            #coverageTest(ngames=1000, epsilon=epsilon, alpha=a, delta=d)
            #ngames=5000, epsilon=0.00, alpha=0.05, delta=0.5
            #writes to 1.accuracyforParams
            p = multiprocessing.Process(target=coverageTest, args=(1000, epsilon, a, d))
            jobs.append(p)
            p.start()

            #############################################
            print(f"(alpha,delta) ({a},{d})")
            print("Creating ConfusionMatrix Fixed P=0.5")
            # C1ConfusionMatrix_fromPoolTest(population, ngames=5000, epsilon=epsilon, alpha=a, delta=d)
            prob = np.full(ngames / 3.0, 0.5)  # (0.5,0.5,1.0/ngames)
            plessthanpointfive = np.arange(0.3, 0.65, ngames * 2.0 / 3.0)
            p = multiprocessing.Process(target=C1ConfusionMatrix_fromPoolTest, args=(prob, 5000, epsilon, a, d,f"fixedp{0.5}"))
            jobs.append(p)
            p.start()
            #############################################
            print(f"(alpha,delta) ({a},{d})")
            print("Creating ConfusionMatrix")
            #C1ConfusionMatrix_fromPoolTest(population, ngames=5000, epsilon=epsilon, alpha=a, delta=d)
            #population,ngames=5000, epsilon=0.05, alpha=0.05, delta=0.5
            p = multiprocessing.Process(target=C1ConfusionMatrix_fromPoolTest, args=(population, 5000, epsilon, a, d))
            jobs.append(p)
            p.start()
            while len(jobs) >= poolsize:  # this is my pool
                # check if any are closed.
                for j in jobs:
                    if not j.is_alive():
                        print(f"-removing {p.pid}")
                        jobs.remove(j)
                time.sleep(1)


        print("-------------------------------------------------------------------------------------------------")
    assert False
    for _ in range(1000000):
        p = multiprocessing.Process(target=plotFixedPAB, args=(0.5, ))
        jobs.append(p)
        p.start()
        while len(jobs) >= poolsize:  # this is my pool
            # check if any are closed.
            for j in jobs:
                if not j.is_alive():
                    print(f"-removing {p.pid}")
                    jobs.remove(j)
            time.sleep(1)
    assert False
    #[plotFixedPAB(0.5) for _ in range(1000000)]
    #while True:
    #    plotFixedPAB(0.5)

