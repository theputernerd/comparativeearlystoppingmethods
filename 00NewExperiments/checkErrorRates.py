from wilson import *
import numpy as np
import sklearn

import numpy as np
import matplotlib.pyplot as plt

from sklearn import svm, datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels

def plot_confusion_matrix(y_true, y_pred, classes,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
    classes = classes[unique_labels(y_true, y_pred)]
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return ax



#use this to test how accurate the stopping is
#eg. nVsP plot gives ngames=5 for a 100% winrate. That means that if I have 100% winrate after 5 games then LCB is >0.5
#n=20 for p=75 should give 95% accuracy when predicting p1!=p2
#to prove this I create a binomial dist with prob P containing n samples.
def exactMethodAccuracy(p=0.5,n=10000,alpha=0.05,hypothesis=0.5,ndistributions=10000,verbose=False):
                        #p = p, n = n, alpha = alpha, ndistributions = ndistributions#print("----exactTest")
    #this shows how to confirm the accuracy of a hypothesis.
    #note you are only interested in type 1 errors.
    #i.e. incorrectly declaring the H0 to be False
    wins = np.random.binomial(n, p, ndistributions)
    from scipy.stats import ttest_1samp,binom_test
    reject=0
    type1=0 #reject H0 when it is actually True.
    type2=0 #Fail to reject H0 when it is actually False
    npredictions=0
    i=0
    #print(f"{p},{n},{alpha},{hypothesis},{ndistributions}")
    for win in wins:
        i+=1
        npredictions+=1
        pval = binom_test(win,n,hypothesis,alternative='two-sided') #alternative : {'two-sided', 'greater', 'less'}
        if pval<alpha: #reject H0 i.e. there is less than a 5% chance that the true p=0.5
            reject+=1
            if p==hypothesis:
                type1+=1
        else:
            if p!=hypothesis:
                type2+=1
        if verbose: print(f"trial {i}: T1:{round(type1/npredictions,4)}, \t T2:{round(type2/npredictions,4)}")

    return [type1/npredictions,type2/npredictions]

def exactMethodAccuracyWithDelta(p=0.5,n=10000,alpha=0.05,hypothesis=0.5,ndistributions=10000,verbose=False, delta=0.05):
    lowDraw = 0.5 - delta
    highDraw = 0.5 + delta
    isdraw = (p > lowDraw and p < highDraw)
    #print("----exactTest")
    #this shows how to confirm the accuracy of a hypothesis.
    #note you are only interested in type 1 errors.
    #i.e. incorrectly declaring the H0 to be False
    wins = np.random.binomial(n, p, ndistributions)
    from scipy.stats import ttest_1samp,binom_test
    reject=0
    type1=0 #reject H0 when it is actually True.
    type2=0 #Fail to reject H0 when it is actually False
    npredictions=0
    i=0
    #print(f"{p},{n},{alpha},{hypothesis},{ndistributions}")
    for win in wins:
        i+=1
        npredictions+=1
        pval = binom_test(win,n,hypothesis,alternative='two-sided') #alternative : {'two-sided', 'greater', 'less'}
        if pval<alpha: #reject H0 i.e. there is less than a 5% chance that the true p=0.5
            reject+=1
            if p==hypothesis:
                type1+=1
        else:
            if not isdraw:
                type2+=1
        if verbose: print(f"trial {i}: T1:{round(type1/npredictions,4)}, \t T2:{round(type2/npredictions,4)}")

    return [type1/npredictions,type2/npredictions]

def incrementalMethodAccuracy(trueP=0.5, blockSize=30, target_alpha=0.05, nTrials=100, astarmin=None, verbose=False, maxgamelength=500,
                              delta=0.05,increment=False):
    #with n=30 every win value is the outcome of 30 games
    #drawTol is the range which the players are considered equal
    if astarmin==None:
        dif=0
        astarmin=target_alpha
    else:
        dif = target_alpha - astarmin
    alpha_star=astarmin #target_alpha
    b = int(maxgamelength/blockSize)-1  # How many blocks?
    decreaseby = dif/b #this is how much to decrease alpha by each block.

    lowDraw= 0.5 - delta
    highDraw= 0.5 + delta
    isdraw=(trueP>=lowDraw and trueP<=highDraw)
    from scipy.stats import ttest_1samp,binom_test
    reject=0
    type1=0 #reject H0 when it is actually True.
    type2=0 #Fail to reject H0 when it is actually False
    npredictions=0
    correct=0
    stoppedafter=[]
     #print(f"{p},{n},{alpha},{hypothesis},{ndistributions}")
    for i in range(1, nTrials):
        wins = np.random.binomial(blockSize, trueP, int(maxgamelength / blockSize)) #have 1000 games int total with results sampled every n
        npredictions += 1
        stopcalled = False
        totalwins = 0
        totalgames = 0
        increments=0
        for win in wins:
            totalwins+=win
            totalgames+=blockSize
            if increment:
                alpha_star= target_alpha - increments * decreaseby
            else:
                alpha_star=astarmin
            pval = binom_test(totalwins,totalgames,0.5,alternative='two-sided') #alternative : {'two-sided', 'greater', 'less'}
            if pval<alpha_star: #reject H0
                reject+=1
                if totalwins/totalgames>0.5 and trueP>0.5:
                    #then it got predicted correctly
                    correct+=1
                elif totalwins/totalgames<0.5 and trueP<0.5:
                    #then it got predicted correctly
                    correct+=1
                elif trueP==0.5: #p actually equals 0.5 so it was incorrectly stopped
                    type1+=1
                stopcalled = True
                    #print(f"type1:{type1}")
                break
            else:
                    pass
                    #if p!=hypothesis:
                    #    type2+=1
            increments+=1

        if not stopcalled:
            if trueP!=0.5:
            #if not isdraw: #trueP != hypothesis:
                #If I didn't stop and the range is within +=drawTol then its not an error.
                type2 += 1
                #print(f"type2:{type2}")

        stoppedafter.append(totalgames)
        if verbose:
            print(f"trial {i}: T1:{type1}_{round(type1/npredictions,4)}, \t T2:{type2}_{round(type2/npredictions,4)},\t{np.mean(stoppedafter)}")
    return [type1/npredictions,type2/npredictions,np.mean(stoppedafter)]

def wilsLCBAccuracy(p=0.5,n=10000,alpha=0.05,ndistributions=100000):
    print("----wilsonTest")
    #this shows how to confirm the accuracy of a hypothesis.
    #note you are only interested in type 1 errors. i.e. incorrectly declaring the H0 to be False
    wins = np.random.binomial(n, p, ndistributions)
    reject=0
    type1=0 #reject H0 when it is actually True.
    type2=0 #Fail to reject H0 when it is actually False
    npredictions=0
    print(f"Obtaining Wilson Single sided Confidence Interval and testing H0:p={0.5}")
    for win in wins:
        npredictions+=1
        l, u, m = wils_int(win, n, alpha=alpha, twosided=True)
        if l>0.5: #reject H0 i.e. there is less than a 5% chance that the true p=0.5
            reject+=1
            if p==0.5:
                type1+=1
        elif u<0.5: #NB if just l then twosdided=False
            reject+=1
            if p==0.5:
                type1+=1
        else:
            if p!=0.5:
                type2+=1
    print(f"Type 1 Error:{round(type1/npredictions,5)}")
    print(f"Type 2 Error:{round(type2/npredictions,5)}")

from prettytable import PrettyTable

def howmanyNforMaxType2SeqMethod(trueP=0.5, blockSize=30, maxType2=0.05, nTrials=100, astarmin=None, verbose=False, maxgamelength=500,
                                 delta=0.05, increment=False, startat=1, maxgamesToTry=2000):
    #Returns how many n to achieve a maximum type 2 error rate of maxType2.
    t = PrettyTable(["P","N","T1","T2"])
    #print(t)
    for n in range(startat, maxgamesToTry):
        #check if n and wins would be whole numbers to get this p. p=x/n
        #for example if I am checking for 90% is 10 games enough?
        x=n*trueP
        #if not x.is_integer():
        #    continue
        t1,t2=incrementalMethodAccuracy(trueP=trueP, blockSize=blockSize, target_alpha=target_alpha, nTrials=nTrials, astarmin=astarmin, verbose=verbose, maxgamelength=n,
                              delta=delta,increment=increment)
        t.add_row([f"{trueP}",f"{n}", f"{round(t1,4)}", f"{round(t2,4)}"])
        print(t)
        if t2>maxType2:
            continue
        else:
            #print(f"{n}")
            return n
    return None

def howmanyNforMaxType2(p=0.7,maxType2=0.05,method=exactMethodAccuracy,alpha=0.05,maxgamesToPlay=2000,ndistributions=10000,startat=1):
    #Returns how many n to achieve a maximum type 2 error rate of maxType2.
    t = PrettyTable(["P","N","T1","T2"])
    #print(t)
    for n in range(startat,maxgamesToPlay):
        #check if n and wins would be whole numbers to get this p. p=x/n
        #for example if I am checking for 90% is 10 games enough?
        x=n*p
        #if not x.is_integer():
        #    continue
        t1,t2=method(p=p,n=n,alpha=alpha,ndistributions=ndistributions)
        t.add_row([f"{p}",f"{n}", f"{round(t1,4)}", f"{round(t2,4)}"])
        print(t)
        if t2>maxType2:
            continue
        else:
            #print(f"{n}")
            return n
    return None

def plotProbVn(probvals, nvals,labels=[], ax=None):
    if ax==None:
        fig = plt.figure(figsize=(16.0, 16.0))
        ax = fig.add_subplot(1, 1, 1)
    minx = 0.5
    ax.set_xlim(minx, 1.0)
    major_xticks = np.arange(minx, 1, .1)
    ax.xaxis.set_ticks(major_xticks)
    minor_xticks = np.arange(minx, 1, .01)
    ax.xaxis.set_ticks(minor_xticks, minor=True)
    ax.grid(b=True, which='major', color='k', linestyle='-', alpha=0.9)
    ax.grid(True, which="minor", ls="--", alpha=0.5)

    start, end = ax.get_xlim()
    for idx, _ in enumerate(probvals):
        ax.semilogy(probvals[idx], nvals[idx])
    ax.set_xlabel(r"Probability of winning $p$")
    ax.set_ylabel(r"Number of games $n$")

    ax.legend(labels)
    ax.set_xlim(0.5,1.0)
    ax.set_ylim(0.5, 2000)

    plt.ylim(10 ** 0)
    #plt.savefig(f"MinSampleSizeFor_{1-alpha}_conf_stopping_LCB.png")
    #plt.draw()
#print(exactMethodAccuracy(p=1,n=6,alpha=0.05,hypothesis=0.5,ndistributions=10000))


import csv

def generateNewDataForFindingOutMinNforAccuratePredictions(alpha=0.05,ndistributions=1000,ps=np.arange(1.0,0.51,-0.01),startfrom=1):
    ######NOT This is designed to count backwards, cause it starts the next count at the max of the previous one.
    nes=[]
    plotps=[]
    n=startfrom
    lastn=0
    for p in ps:
        n=howmanyNforMaxType2(p=round(p,2),maxType2=0.05,method=exactMethodAccuracy,alpha=alpha,startat=n,ndistributions=ndistributions)
        if n!=None:
            print("-------------------")
            lastn=n
            with open('minSampleSize.csv', 'a',newline="") as writeFile:
                writer = csv.writer(writeFile)
                val=[p,n,ndistributions,alpha]
                writer.writerow(val)
            nes.append(n)
            plotps.append(p)
            print("-------------------")
        else:
            n=lastn
#colate and plot up the csv data.
def load():
    theplots = {}
    p_dict = {}
    with open('minSampleSize.csv', 'r', newline="") as readFile:
        lines = csv.reader(readFile)
        for l in lines:
            [p, n, ndistributions, alpha] = l
            [p, n, ndistributions, alpha] = [round(np.float(p), 2), round(np.float(n), 2),
                                             round(np.float(ndistributions), 2), round(np.float(alpha), 2)]
            if alpha not in theplots:
                theplots[alpha] = {}
                thisplotdict = theplots[alpha]
            else:
                thisplotdict = theplots[alpha]

            if p not in thisplotdict:
                thisplotdict[p] = [p, n, ndistributions, alpha]
                singleplot = thisplotdict[p]
            else:
                singleplot = thisplotdict[p]
                singleplot[0] = (singleplot[2] * singleplot[0] + ndistributions * p) / (singleplot[2] + ndistributions)
                singleplot[1] = (singleplot[2] * singleplot[1] + ndistributions * n) / (singleplot[2] + ndistributions)
                singleplot[2] += ndistributions
    return theplots

def Plot(theplots):

    legend=[]
    xes=[]
    yes=[]
    for alpha,plotdata in theplots.items():
        nes = []
        plotps = []

        legend.append(r"$\beta="+str(alpha)+r"$")
        for value in plotdata.values():
            [p, n, ndistributions,alpha]=value

            plotps.append(p)
            nes.append(n)
        xy=list(zip(plotps,nes))
        xy=sorted(xy)
        x,y=zip(*xy)
        xes.append(x)
        yes.append(y)
        print("Confirm predictions")
        p, n, samples,alpha = plotdata[0.55]

        print(f"-----Type2 error should be <0.05, for: p:{p}, n:{n}, alpha:{alpha} ")
        print(exactMethodAccuracy(p, n, alpha, ndistributions=10000))
        print("-------------------------------")


    fig = plt.figure(figsize=(16.0, 16.0))

    ax = fig.add_subplot(1, 1, 1)
    ax.set_title(r"Min $n$ required for desired Type II error $\beta$")
    plotProbVn(xes,yes,legend,ax=ax)
    fig.savefig("nGamesForType2Error.png")
    fig.savefig("nGamesForType2Error.eps")

    plt.show()


def vary_alpha_to_get_confidence(p=0.5, maxTypeError=0.05, target_alpha=0.05, alphastar_start=0.01,
                                 maxgamelength=2000, nTrials=1000, blocksize=50,verbose=False):
    #Returns how many n to achieve a maximum type 2 error rate of maxType2.

    t = PrettyTable(["Target alpha","a*","T1","T2"])
    #print(t)
    alphainc=0.001
    smallestalpha=1e-6
    alphas=np.arange(alphastar_start,smallestalpha,-alphainc)
    for a in alphas:
        #check if n and wins would be whole numbers to get this p. p=x/n
        #for example if I am checking for 90% is 10 games enough?

        t1,t2,ngames=incrementalMethodAccuracy(trueP=p, blockSize=blocksize, target_alpha=target_alpha, nTrials=nTrials, maxgamelength=maxgamelength
                                               , verbose=False,astarmin=a,increment=False)
        t.add_row([f"{target_alpha}", f"{round(a,4)}", f"{round(t1,4)}", f"{round(t2,4)}"])
        if verbose:
            print(t)
        if p==0.5:
            errortotest=t1
        else:
            errortotest=t2
        if errortotest>maxTypeError:
            continue
        else:
            #print(f"{n}")
            return a
    return None

############################## STEP 0. Prove that my code works by solving a fixed problem
# note there is not enough games to to converge to the confidence so you get high Type2
"""https://machinelearningmastery.com/statistical-power-and-power-analysis-in-python/
When interpreting statistical power, we seek experiential setups that have high statistical power.
Low Statistical Power: Large risk of committing Type II errors, e.g. a false negative.
High Statistical Power: Small risk of committing Type II errors

Power = 1 - Type II Error
Pr(True Positive) = 1 - Pr(False Negative)
 β is usually called type II error and 1−β is called power. https://stats.stackexchange.com/questions/156973/relation-between-power-and-sample-size-in-a-binomial-test
α = P{type I error}
= P{reject H0 when H0 is true},
β = P{type II error}
= P{fail to reject H0 when H0 is false}.
https://www.mobt3ath.com/uplode/book/book-43792.pdf
An upper bound for α is a significance level of the test procedure. Power of
the test is defined as the probability of correctly rejectingthe null hypothesis
when the null hypothesis is false, i.e.,
Power = 1 − β
= P{reject H0 when H0 is false}.
With a fixed sample size a typical approach is
to avoid a type I error but at the same time to decrease a type II error
so that there is a high chance of correctly detecting a drug effect when
the drugis indeed effective
Typically, when the sample size is fixed, α
decreases as β increases and α increases as β decreases. The only approach
to decrease both α and β is to increase the sample size. Sample size is
usually determined by controllingboth type I error (or confidence level)
and type II error (or power).

Since a type I error is usually considered to be a more important and/or
serious error which one would like to avoid, a typical approach in hypothesis
testingis to control α at an acceptable level and try to minimize β by
choosingan appropriate sample size. In other words, the null hypothesis
can be tested at pre-determined level (or nominal level) of significance with
a desired power. This concept for determination of sample size is usually
referred to as power analysis for sample size determination

p29.. For determination of sample size based on power analysis, the investigator is required to specify the following information. First of all, select
a significance level at which the chance of wrongly concluding that a difference exists when in fact there is no real difference (type I error) one is
willingto tolerate. Typically, a 5% level of significance is chosen to reflect
a 95% confidence regarding the unknown parameter. Secondly, select a desired power at which the chance of correctly detectinga difference when
the difference truly exists one wishes to achieve. A conventional choice of
power is either 90% or 80%. Thirdly, specify a clinically meaningful difference. In most clinical trials, the objective is to demonstrate effectiveness
and safety of a drugunder study as compared to a placebo. Therefore, it
is important to specify what difference in terms of the primary endpoint is
considered of clinical or scientifical importance. Denote such a difference by
. If the investigator will settle for detecting only a large difference, then
fewer subjects will be needed. If the difference is relatively small, a larger
study group (i.e., a larger number of subjects) will be needed. Finally, the
knowledge regarding the standard deviation (i.e., σ) of the primary endpoint considered in the study is also required for sample size determination.
A very precise method of measurement (i.e., a small σ) will permit detection of any given difference with a much smaller sample size than would be
required with a less precise measurement.

You can use wilson for confidence testing see https://www.itl.nist.gov/div898/handbook/prc/section2/prc241.htm

"""
if False:
    print(f"---------------p=0.50")
    print(f"--- With only 100 games, the hypothesis is rarely rejected, so the Type 1 error is lower than projected")
    print(exactMethodAccuracy(0.50, 100, .05, ndistributions=50000))
    #[0.03588, 0.0]
    print(f"--- Even with 20000 games, Type 1 error stays below 0.05")
    print(exactMethodAccuracy(0.50, 20000, .05, ndistributions=50000))
    #[0.04932, 0.0]
    print(f"---------------p=0.55")
    print(f"100 games, shows not enough games to make a decision so Type 2 errors are high.")
    print(f"{exactMethodAccuracy(0.55, 100, .05, ndistributions=50000)}")
    # [0.0, 0.86196]
    print(f"1302 games is approximately enough games to make predictions to reduce type 2 errors to 0.05 for p=0.55.")
    print(exactMethodAccuracy(0.55, 1302, .05, ndistributions=50000))
    # [0.0, 0.0504]
    print(f"If you do infinite games you will make always make prediciton, so Type2 tends towards 0. "
          f"\neg. 20000 games is approximately enough games to reduce type 2 errors to 0 for p=0.55."
          f"\nif you want to control type2 then limit number of games. ")
    print(exactMethodAccuracy(0.55, 20000, .05, ndistributions=50000))
    #[0.0, 0.0]
    if False:
        maxGames=3000
        truePs=[0.5,0.55,0.60]
        for trueP in truePs:
            t1,t2=exactMethodAccuracy(trueP, maxGames, .05, ndistributions=20000)
            print(f"Truep={trueP}")
            print(t1,t2)
            print('---------------------------')
print(f"---------------------------------------------------------------------------------------------------")

wilsLCBAccuracy(p=0.5,n=100,alpha=0.05)
input("Press Enter to continue...")
################################STEP 0.1 Generate the selection plot.
####TODO: What does the alpha_star plot look like with changing p to obtain the required accuracy
ALLON=False
varyingPType2ErrorPlot=True
if varyingPType2ErrorPlot:
    print(
        f"-------------------------0.1.Generate the plot of number of games for varying P to give required Type 2 Error rate.")
    generateNewDataForFindingOutMinNforAccuratePredictions(alpha=0.01,ndistributions=15000,ps=np.arange(1.0,0.51,-0.01),startfrom=1)
    theplots=load()
    Plot(theplots)
    input("Press Enter to continue...")
    #maxGames = 2000
print(f"-------------------------1. Decide on block size, Ntrials,target alpha, and p_min  and use a high maxGames for TYPE 1 Error, .")
############################# nTrials is for testing accuracy of the prediction
print(f"target_alpha:{target_alpha}. This is the errror rate or 1-confidence")
p_min=0.55
blocksize = 100
print(f"blocksize:{blocksize}. How many games to play before checking")
maxGames = 5000
target_alpha=0.05
print(f"p_min:{p_min} This is the value closest to 0.5 that the desired confidence is required.")
nTrials=1000
print(f"nTrials:{nTrials}. nTrials is for testing accuracy of the prediction")
input("Press Enter to continue...")

print(f"-------------------------2. Vary alpha ->(alpha*) with p=0.5 until the required Type 1 error is below target_alpha using the sequential stopping method")
astarmin=0.008
getAlphastar=True
if getAlphastar or ALLON:
    astarmin=vary_alpha_to_get_confidence(p=0.5, maxTypeError=target_alpha, target_alpha=target_alpha, nTrials=nTrials, maxgamelength=maxGames, blocksize=blocksize,verbose=True)
print(f"astar:{astarmin}")
input("Press Enter to continue...")
#############################   STEP 4. Get maxGames. How many games to have <alpha TYPE 2 Errors for the lowest P I need. use full mthod
print(f"-------------------------3. "
      f"Run the sequential method for p_min and vary the number of games to obtain the desired Type 2 Error level. Or take it from the plot.")

maxGames = 1828  # I already know this now. for 0.55 Too many and we increase chance of error cause of convergence. too few and we dont get predictions
if True or ALLON:
    nTrials=nTrials*2
    maxGames = howmanyNforMaxType2(p=round(p_min, 2), maxType2=target_alpha, method=exactMethodAccuracy, alpha=astarmin,startat=1738, ndistributions=nTrials)

    ####TODO: Get this sorted for the sequential method!.
    #maxGames = howmanyNforMaxType2SeqMethod(trueP=round(p_min, 2), blockSize=blocksize, maxType2=target_alpha, alpha=astarmin, startat=1738,
    #                                        nTrials=nTrials, verbose=False, maxgamesToTry=maxGames * 2)
#trueP=0.5, blockSize=30, maxType2=0.05, nTrials=100, astarmin=None, verbose=False, maxgamelength=500,
#                              delta=0.05,increment=False,startat=1,maxgamesToPlay=2000
# print("--------------------------------")
# maxGames = howmanyNforMaxType2(p=round(p_min, 2), maxType2=target_alpha, method=exactMethodAccuracy, alpha=target_alpha, startat=1300,ndistributions=nTrials)
print(f"maxGames:{maxGames}")
input("Press Enter to continue...")
#############################   STEP 5. Check the performance with the parameters I have.
print(f"-------------------------4. Final Test ")
truePs=[0.5,0.55,0.60]
for trueP in truePs:
    t1,t2,ngames=incrementalMethodAccuracy(trueP=trueP, blockSize=blocksize, target_alpha=target_alpha, nTrials=nTrials, maxgamelength=maxGames
                                           , verbose=False, astarmin=astarmin)
    print(f"Truep={trueP}")
    print(t1,t2,ngames)
    print('---------------------------')



#print(vary_alpha_to_get_confidence(p=0.5, maxTypeError=0.05, target_alpha=0.05, nTrials=5000,maxgamelength=1300))
#incrementalMethodAccuracy(0.50, 100, 0.009, nTrials=5000, verbose=True,maxgamelength=1300)
##print(incrementalMethodAccuracy(0.50, 100, 0.008, nTrials=5000, verbose=True,maxgamelength=2000))
#print(exactMethodAccuracy(0.50, 500, .05, ndistributions=5000,verbose=False))
"""
with alpha*=0.008, maxgamelength=2000, averaged over 5000 trials, with a blocksize of 100, Type1 Error is 0.049 for p=0.5  
"""
#######
theplots=load()
Plot(theplots)
