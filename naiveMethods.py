import math
from lib import *

g = 300

class n_games:
    #This test plays g games and chooses a winner after those many games are played based solely on the number of wins.
    #p1>p2 or p2>p1 or p1=p2

    name=f"bestof_{g}_games"
    desc=f"plays {g} games and prediction based on max wins. Predicts either" \
         f"player 1, 2 or 0 for draw. It keeps drawn games in the dataset."
    finished=False
    @staticmethod
    def reset():
        n_games.finished = False
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
            c = conf_stopped_struct(n_games.name, p1.nWins, p1.nWins, p2.nWins, drawsP.nWins, ngames, best_predict, best_actual,best_actual==best_predict,-1.0)
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
    def reset():
        perc_after_n_games.finished = False

    @staticmethod
    def start( p1,p2,drawsP,best_actual):
        #this condition only plays 300 games then decides only if one agent is 55% better.
        finish = False
        c = None
        z = None
        if perc_after_n_games.finished:  # Already made a prediction
            return finish, z, c
        totGames = p1.nWins + p2.nWins +drawsP.nWins

        if (totGames)<g: #needs 300 wins or losses
            return finish, z, c
        if (totGames)>g: #needs 300 wins or losses
            return finish, z, c

        finish=True
        nGames=p1.nWins + p2.nWins
        p1winrate=p1.nWins/nGames
        p2winrate=p2.nWins/nGames

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
            c = conf_stopped_struct(perc_after_n_games.name, p1winrate, p1.nWins, p2.nWins, drawsP.nWins, totGames, best_predict, best_actual,best_actual==best_predict,-1.0)
            perc_after_n_games.finished=True

        return finish, z, c
