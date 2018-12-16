
import matplotlib as mpl
#mpl.use('Agg')
import numpy as np
import random
from lib import player,drawOk
playerPoolDist = random.uniform(1.5, 1.9)
from wilson import wils_int
from lib import shouldIStop,game
from bayes import bayesian_U
tests = [bayesian_U, wils_int]
import time
import os
import matplotlib.pyplot as plt


def playOneGame(g, results):
    winner = g.playGame()
    # winner = drawsP #change remove me.
    winner.nWins += 1
    return results


def testAccuracy(nGames,p1winrate,p2winrate=None,drawRate=None,trials=1,epsilon=0.01,delta=0.5):
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
                winner,method= shouldIStop(1, p1L, p1U, mean, n, delta=delta, epsilon=epsilon)
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
                winner,method= shouldIStop(3, p1L, p1U, mean, n, delta=delta, epsilon=epsilon)
                if winner!=0 and not wilsPredicted:
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
                winner,method= shouldIStop(1, p1L, p1U, mean, n, delta=delta, epsilon=epsilon)
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
                winner,method= shouldIStop(3, p1L, p1U, mean, n, delta=delta, epsilon=epsilon)
                if winner != 0 and not baysPredicted:
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
from scipy.stats import truncnorm,uniform
import scipy
def get_truncated_normal(mean=0, sd=1, low=0, upp=10):
    return uniform()
    return truncnorm(
        (low - mean) / sd, (upp - mean) / sd, loc=mean, scale=sd)

def choosefromPoolTest(population,ngames=5000, epsilon=0.05, alpha=0.05, delta=0.5):
    assert False
    s=population.rvs(ngames)
    fig3 = plt.figure(figsize=plt.figaspect(0.5))
    ax3 = fig3.add_subplot(1, 1, 1)
    ##This test will select from a pool of players normally distributed around 0.5
    count, bins, ignored = plt.hist(s, 100, normed=False)
    #plt.plot(bins, 1 / (sigma * np.sqrt(2 * np.pi)) *np.exp( - (bins - mu)**2 / (2 * sigma**2) ),linewidth=2, color='r')
    ax3.set_title("Population distribution for testing prediction")
    ax3.set_ylabel("Quantity")
    ax3.set_xlabel("Probability player A is better than player B")
    fig3.savefig(f"failureTest/populationHist_eps{epsilon}_alpha{alpha}.png", format='png')

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
                winner, method = shouldIStop(1, p1L, p1U, mean, n, delta=delta, epsilon=0)  #no threshold for lcb only
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
                winner, method = shouldIStop(3, p1L, p1U, mean, n, delta=delta, epsilon=epsilon)
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



            if not baysPredicted:
                #########################BAYES CONDITION 1

                p1L, p1U, mean = bayesian_U(p1.nWins, n, alpha)
                winner, method = shouldIStop(1, p1L, p1U, mean, n, delta=delta, epsilon=0)
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
                winner, method = shouldIStop(3, p1L, p1U, mean, n, delta=delta, epsilon=epsilon)
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
        if pab<0.5-delta:
            wthisbin=wpredictiongrid['Pab<0.5-delta']
            bthisbin=bpredictiongrid['Pab<0.5-delta']

        elif 0.5-delta<=pab and pab<0.5:
            wthisbin=wpredictiongrid['0.5-delta<=Pab<0.5']
            bthisbin=bpredictiongrid['0.5-delta<=Pab<0.5']

        elif 0.5<pab and pab<=0.5+delta:
            wthisbin=wpredictiongrid['0.5<Pab<=0.5+delta']
            bthisbin=bpredictiongrid['0.5<Pab<=0.5+delta']
        elif pab==0.5:
            wthisbin=wpredictiongrid['Pab==0.5']
            bthisbin=bpredictiongrid['Pab==0.5']

        elif pab>0.5+delta:
            wthisbin=wpredictiongrid['Pab>0.5+delta']
            bthisbin=bpredictiongrid['Pab>0.5+delta']
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
            line += "|{0:<7}|{1:<7}|{2:<7}|{3:<7}".format("B", "Draw", "A", "Ngames")
            print(line)

            for key,v in wpredictiongrid.items():
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


    with open(f"failureTest/accuracyTest_eps{epsilon}_alpha_{alpha}_predicMargin={delta}.txt", "w") as f:

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
        line += "|{0:<7}|{1:<7}|{2:<7}|{3:<7}".format("B", "Draw", "A", "Ngames")
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
        line += "|{0:<7}|{1:<7}|{2:<7}|{3:<7}".format("B", "Draw", "A", "Ngames")
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

def coverageTest(ngames=5000, epsilon=0.00, alpha=0.05, delta=0.5):
    #Iterates over a series of p values and records the coverage for that value.
    assert False


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
    s=np.arange(0.20,0.8,0.02)
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
            if p1.pWin>0.5+delta:
                best_actual = 1
            elif p1.pWin<0.5-delta:
                best_actual = 2
            else:
                best_actual=3

        #p=0.46193
        #p1 = player(p)
        #p2 = player(1 - p)
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

                if not wilsPredicted and n>7:
                    #########################WILSON CONDITION 1
                    #p1.pWin = 0.57
                    #p1.nWins = 823
                    #n = 1507
                    p1L, p1U, mean = wils_int(p1.nWins, n, alpha)
                    p1L=np.round(p1L,3)
                    p1U = np.round(p1U  , 3)
                    mean = np.round(mean, 3)

                    winner, method = shouldIStop(1, p1L, p1U, mean, n, delta=delta,
                                                 epsilon=epsilon)  #no threshold for lcb only
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
                    p1L = np.round(p1L, 3)
                    p1U = np.round(p1U, 3)
                    mean = np.round(mean, 3)

                    winner, method = shouldIStop(3, p1L, p1U, mean, n, delta=delta, epsilon=epsilon)
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

                    winner, method = shouldIStop(1, p1L, p1U, mean, n, delta=delta, epsilon=epsilon)
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
                    p1L = np.round(p1L, 3)
                    p1U = np.round(p1U, 3)
                    mean = np.round(mean, 3)
                    winner, method = shouldIStop(3, p1L, p1U, mean, n, delta=delta, epsilon=epsilon)
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

    ax.plot(wilX,wilY,'x')
    ax.set_title(f"Accuracy using Wilson.alpha={alpha} delta={delta}")
    ax.set_xlabel(f"True probability Pab")
    ax.set_ylabel(f"Prediction accuracy")
    fig1.savefig(f"failureTest/wilsoncoverage_alpha={alpha}_epsilon={epsilon}_delta={delta}.png",format="png")
    fig2 = plt.figure(figsize=plt.figaspect(0.5))
    ax2 = fig2.add_subplot(1, 1, 1)
    ax2.set_title(f"Accuracy using Bayesian-U. alpha={alpha} delta={delta}")
    ax2.set_xlabel(f"True probability Pab")
    ax2.set_ylabel(f"Prediction accuracy")
    ax2.plot(bayX, bayY,'x')
    fig2.savefig(f"failureTest/bayescoverage_alpha={alpha}_epsilon={epsilon}_delta={delta}.png",format="png")
    plt.show()

def getNgamesToPredicefixedpab(ntrials,alpha, delta, p1w,nplayed):
    #returns xyz values for accuracy. y is accuracy, x,y is alpha delta
    #note nplayed only is used to determine probability of winning
    Bcorrect = 0
    Wcorrect = 0
    wAvGamesToPredic=[]
    bAvGamesToPredic=[]
    wcorrect = []
    bcorrect = []
    p=p1w/nplayed
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

    #p1L, p1U, mean = wils_int(p1w, nplayed, alpha)
    #p1L = np.round(p1L, 3)
    #p1U = np.round(p1U, 3)
    #mean = np.round(mean, 3)
    n=0#nplayed
    #stop, winner = shouldIStop(1, p1L, p1U, mean, n, delta=delta, epsilon=0)
    for i in range(ntrials):
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

            if not wilsPredicted : #todo: remove this.
                #########################WILSON CONDITION 1
                # p1.pWin = 0.57
                # p1.nWins = 823
                # n = 1507
                p1L, p1U, mean = wils_int(p1.nWins, n, alpha)
                p1L = np.round(p1L, 3)
                p1U = np.round(p1U, 3)
                mean = np.round(mean, 3)

                stop, winner = shouldIStop(1, p1L, p1U, mean, n, delta=delta, epsilon=epsilon)  # no threshold for lcb only
                method=1
                if stop:  ##condition1
                    wilsPredicted = True
                    # now to see if prediction is correct.
                    if int(1) != int(1):
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
                                                p1U - p1L, "or Draw"]
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
                            print(f"wils {winner} failed. {storedResult}")

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
                method=3
                if stop and not wilsPredicted:
                    wilsPredicted = True
                    # now to see if prediction is correct.
                    if int(2) != int(2):
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
                                                p1U - p1L, "or Draw"]
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
                            print(f"wils{winner} failed. {storedResult}")

            if not baysPredicted:
                #########################BAYES CONDITION 1

                p1L, p1U, mean = bayesian_U(p1.nWins, n, alpha)

                stop, winner = shouldIStop(1, p1L, p1U, mean, n, delta=delta, epsilon=epsilon)
                if stop:
                    baysPredicted = True
                    # now to see if prediction is correct.
                    if int(1) != int(1):
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
                            p1.pWin, p1.nWins, n, True, 3, p1L, p1U, mean, winner, best_actual, p1U - p1L)
                            bayesresults = storedResult
                            Bcondition1 = storedResult
                        else:
                            storedResult = [p1.pWin, p1.nWins, n, False, method, p1L, p1U, mean, winner, best_actual,
                                            p1U - p1L]
                            bayesresults = storedResult
                            Bcondition1 = storedResult
                            print(f"bayes{winner} failed. {storedResult}")

                #########################BAYES CONDITION 2
                p1L, p1U, mean = bayesian_U(p1.nWins, n, alpha / 2)
                p1L = np.round(p1L, 3)
                p1U = np.round(p1U, 3)
                mean = np.round(mean, 3)
                stop, winner = shouldIStop(3, p1L, p1U, mean, n, delta=delta, epsilon=epsilon)
                if stop and not baysPredicted:
                    baysPredicted = True
                    if int(3) != int(3):
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
                                                p1U - p1L, "or Draw"]

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
                            print(f"bayes{winner} failed. {storedResult}")
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

def SamplesForPAB(p1W,nPlayed=1):
    #alpha=numpy.arange(0.10,0.01,-0.01)
    #delta=numpy.arange(0.10,0.01,-0.01)
    #alpha = numpy.arange(0.01,0.11,0.01) #[1-0.95,1-0.975,1-0.99] #numpy.arange(.01, 0.205, 0.01) #TODO Add more fidelity make incr smaller
    #delta = numpy.arange(0.01,0.11,0.01)#[0.05,0.075,0.1]#numpy.arange(.01, 0.205, 0.01)
    alpha = numpy.arange(.01, 0.101, 0.02)  # TODO Add more fidelity make incr smaller
    delta = numpy.arange(.05, 0.201, 0.02)
    zWaccuracy=[]
    zWnum=[]
    zBaccuracy = []
    zBnum = []
    x=[]
    y=[]
    z=[]
    n_extrapolatedPts=200
    ngames=20
    elev=30
    az=117
    import csv
    pab=p1W/float(nPlayed)
    print(f"doing {p1W}")
    for a in alpha:
        for d in delta:
            a=round(a,3)
            d=round(d,3)
            #print(f"({a},{d})")
            [W,B]=getNgamesToPredicefixedpab(ntrials=10,alpha=a, delta=d, p1w=p1W,nplayed=nPlayed)
            x.append(a)
            y.append(d)
            zWaccuracy.append(W[0])
            zWnum.append(W[1])
            zBaccuracy.append(B[0])
            zBnum.append(B[1])
            with open('failureTest/ngamesforPAB.csv','a',newline='') as csvfile:
                spamwriter = csv.writer(csvfile, delimiter=',')
                line=[pab,ngames,round(a,4),round(d,4),W[0],W[1],B[0],B[1]]
                spamwriter.writerow(line)
    print("written to failureTest/ngamesforPAB.csv")
    """
    from matplotlib import cm
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import axes3d
    #############################################Wilson accuracy

    fig = plt.figure(figsize=plt.figaspect(0.5))
    xi,yi,zi=interpretXYZ(x,y,zWaccuracy,n_extrapolatedPts)
    ax = fig.add_subplot(1, 1, 1, projection='3d')
    cs1 = ax.contourf(xi, yi, zi, 500, linewidths=1,cmap=cm.jet)
    ax.invert_yaxis()
    ax.set_xlabel('alpha')
    # ax.set_xlim(0, maxNgames)
    ax.set_ylabel('delta')
    ax.set_zlabel('Accuracy')
    ax.view_init(elev=elev, azim=az)
    ax.set_title(f"Accuracy using Wilson Pab={pab}")
    plt.colorbar(cs1,ax=ax)
    fig.savefig(f"cont3d_Wilsonp={pab}.png", format='png')
    fig.show()
    plt.show()


    #############################################Bayes
    fig2 = plt.figure(figsize=plt.figaspect(0.5))
    xi,yi,zi=interpretXYZ(x,y,zBaccuracy,n_extrapolatedPts)
    ax2 = fig2.add_subplot(1, 1, 1, projection='3d')
    cs2 = ax2.contourf(xi, yi, zi, 500, linewidths=1, cmap=cm.jet)
    ax2.invert_yaxis()
    ax2.set_xlabel('alpha')
    # ax.set_xlim(0, maxNgames)
    ax2.set_ylabel('delta')
    ax2.set_zlabel('Accuracy')
    ax2.view_init(elev=elev, azim=az)
    ax2.set_title(f"Accuracy using Bayes-U Pab={pab}")

    plt.colorbar(cs2, ax=ax)
    fig2.savefig(f"cont3d_Bayesp={pab}.png", format='png')

    fig2.show()
    plt.show()

    #############################################Wilson nGames

    fig3 = plt.figure(figsize=plt.figaspect(0.5))
    xi, yi, zi = interpretXYZ(x, y, zWnum, n_extrapolatedPts)
    ax3 = fig3.add_subplot(1, 1, 1, projection='3d')
    cs1 = ax3.contourf(xi, yi, zi, 500, linewidths=1, cmap=cm.jet)
    ax3.invert_yaxis()
    ax3.set_xlabel('alpha')
    # ax.set_xlim(0, maxNgames)
    ax3.set_ylabel('delta')
    ax3.set_zlabel('Ngames to decision')
    ax3.view_init(elev=elev, azim=az)
    ax3.set_title(f"NGames to decision Wilson Pab={pab}")
    plt.colorbar(cs1, ax=ax3)
    fig3.savefig(f"nGames_Wilsonp={pab}.png", format='png')
    fig3.show()
    plt.show()
    #############################################Bayes nGames

    fig4 = plt.figure(figsize=plt.figaspect(0.5))
    xi, yi, zi = interpretXYZ(x, y, zBnum, n_extrapolatedPts)
    ax4 = fig4.add_subplot(1, 1, 1, projection='3d')
    cs1 = ax4.contourf(xi, yi, zi, 500, linewidths=1, cmap=cm.jet)
    ax4.invert_yaxis()
    ax4.set_xlabel('alpha')
    # ax.set_xlim(0, maxNgames)
    ax4.set_ylabel('delta')
    ax4.set_zlabel('Ngames to decision')
    ax4.view_init(elev=elev, azim=az)
    ax4.set_title(f"NGames to decision Bayes-U Pab={pab}")
    plt.colorbar(cs1, ax=ax4)
    fig4.savefig(f"nGames_Bayesp={pab}.png", format='png')
    fig4.show()
    plt.show()
    pass
    """
from multiprocessing import Pool
import multiprocessing
if __name__ == '__main__':
    np.random.seed(None)  # changed Put Outside the loop.
    random.seed()
    pvals=[0.50] #could od other values ,0.05,0.25,0.3,0.4,0.45,0.50
    pvals+=pvals
    try:
        os.mkdir("failureTest")
    except FileExistsError:
        pass
    jobs=[]
    poolsize=12
    while True:
            for pv in pvals:
                #res=p.apply_async(SamplesForPAB,pv)
                p = multiprocessing.Process(target=SamplesForPAB, args=(pv,))
                jobs.append(p)
                p.start()
                print(f"starting {p.pid}")
                while len(jobs)>=poolsize: #this is my pool
                    #check if any are closed.
                    for j in jobs:
                        if not j.is_alive():
                            print(f"-removing {p.pid}")
                            jobs.remove(j)
                    time.sleep(1)


        #p.map(SamplesForPAB,pvals)
#        [SamplesForPAB(p) for p in pvals]
        #for p in pvals:
        #    SamplesForPAB(p,nPlayed=1)

    assert False
    fullResult=dict()
    alpha=0.05
    epsilon=0.0
    delta=0.05
    alphaList=[0.1,0.05,0.01,0.001]
    deltaList=[0.1,0.05,0.01]

    #######plt.title(r'$\alpha > \beta$')
    mu, sigma = 0.5, .2  # mean and standard deviation
    population = get_truncated_normal(mu, sigma, 0,1)
    #population=np.random.uniform(0.1,0.9,100)
    for a in alphaList:
        for d in deltaList:
            coverageTest(ngames=1000, epsilon=epsilon, alpha=a, delta=d)
            choosefromPoolTest(population,ngames=1000, epsilon=epsilon, alpha=a, delta=d)
            print("-------------------------------------------------------------------------------------------------")


    pass
