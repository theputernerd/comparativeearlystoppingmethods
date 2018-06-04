from lib import *

class bayes():
    limit=0.70
    minGames=1

    name = f"bayes_limit={limit}"
    desc = f""

    finished = False
    H_A = "p1 is better"
    H_B = "p2 is better"
    H_C = "p1 and p2 are equal"
    D="Event is a draw"
    E="Event is p1 wins."
    #####Priors
    priorStart=0.5
    P_A = priorStart  # probably the hypothesis is true. These are updated.
    P_B = priorStart
    P_C = priorStart
    ##likelihood
    P_E_A = .51  # Likelihood that given event E the hypothesis is true. These were somewhat plucked
    P_E_B = .49
    P_E_C = .50

    p1wins = 0  # this keeps track of the wins I know about, cause I just need to know if the last game was a win.
    p2wins = 0
    draws = 0

    @staticmethod
    def reset():
        #####Priors
        bayes.P_A = bayes.priorStart
        bayes.P_B = bayes.priorStart
        bayes.P_C = bayes.priorStart


        bayes.p1wins = 0  # this keeps track of the wins I know about, cause I just need to know if the last game was a win.
        bayes.p2wins = 0
        bayes.draws = 0
        bayes.finished=False
    @staticmethod
    def start(p1,p2,drawsP,best_actual):
        #https://www.evanmiller.org/how-not-to-sort-by-average-rating.html
        p1Won=True
        finished = False
        c = None
        if bayes.finished:
            return finished, 0, c

        #Now find posterior prob
        if p1.nWins!=bayes.p1wins :
            #then p1 won.
            posA = bayes.P_E_A * bayes.P_A
            posB = bayes.P_E_B * bayes.P_B
            posC = bayes.P_E_B * bayes.P_C
            bayes.p1wins = p1.nWins
        elif p2.nWins!=bayes.p2wins:
            #do update for p2 winning.
            posA = (1-bayes.P_E_A) * (bayes.P_A)
            posB = (1-bayes.P_E_B) * (bayes.P_B)
            posC = (bayes.P_E_B) * (bayes.P_C) #If p1wins the prob of the event given H(B) is no different than for p1
            bayes.p2wins=p2.nWins
        elif drawsP.nWins!=bayes.draws: #then it was a draw
            posA = 0.50 * (bayes.P_A)
            posB = 0.50 * (bayes.P_B)
            posC = 0.50 * (bayes.P_C)
            bayes.draws = drawsP.nWins
            pass
        else:
            assert False #Did you even play a game before calling?
        normSum=posA+posB+posC
        posA=posA/normSum
        posB = posB / normSum
        posC = posC / normSum
        bayes.P_A=posA   #These are the latest values
        bayes.P_B=posB
        bayes.P_C=posC

        lowerBound = 0
        totGames = p1.nWins + p2.nWins + drawsP.nWins

        if bayes.minGames>totGames: #in the case of equal players there might be a large number of draws, so at least give it some good data to decide.
            return finished, lowerBound, c

        if bayes.P_A>bayes.limit:
            best_predict=1
            finished=True
        elif bayes.P_B>bayes.limit:
            best_predict=2
            finished=True

        elif bayes.P_C>bayes.limit:
            best_predict=0
            print("Predicted a draw")

            finished=True




        if finished:
            #print(f"{p1.nWins},{p2.nWins},{lowerBound},{upperBound}")
            totGames = p1.nWins + p2.nWins + drawsP.nWins
            bayes.finished=True
            c=conf_stopped_struct(bayes.name, [bayes.P_A, bayes.P_B,bayes.P_C], p1.nWins, p2.nWins, drawsP.nWins, totGames, best_predict, best_actual, best_actual == best_predict, -1.0)
            if best_actual!=best_predict:
                print(c)
                pass

        return finished,lowerBound,c
