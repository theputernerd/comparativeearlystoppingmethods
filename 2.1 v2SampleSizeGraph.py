import csv
import numpy as np
def interpretXYZ(x,y,z,pts=1000):
    from scipy.interpolate import griddata
    import random
    if len(x) > 0:
        x, y,z = zip(*random.sample(list(zip(x, y,z)), int(len(x) / 1)))
    xi = np.linspace(min(x), max(x),pts)
    yi = np.linspace(min(y), max(y),pts)
    zi = griddata((x, y), z, (xi[None,:], yi[:,None]), method='cubic')
    return xi,yi,zi


xf ='xf'# []
yf ='yf'# []
zWaccuracyf ='zWaccuracyf'# []
zWnumf = 'zWnumf'#[]
zBaccuracyf = 'zBaccuracyf'#[]
zBnumf = 'zBnumf' #[]
nf='nf'
pab=[]
pAB={}
roundTo=3
show=False
with open('failureTest/ngamesforPAB.csv', 'rU') as csvfile:
    reader = csv.reader(csvfile, delimiter=',')
    for row in reader:
        p,n,af, df, zWaf, zWnf, zBaf, zBnf=row
        p=round(float(p),roundTo)
        pab.append(p)
        if p not in pAB:
            pAB[p]={}
            p=pAB[p]
            p[nf]=[]
            p[xf]=[]
            p[yf]=[]
            p[zWaccuracyf]=[]
            p[zBaccuracyf]=[]
            p[zBnumf]=[]
            p[zWnumf]=[]

        else:
            p=pAB[p]
        p[nf].append(float(n))
        p[xf].append(round(float(af),roundTo))
        p[yf].append(round(float(df),roundTo))
        p[zWaccuracyf].append(float(zWaf))
        p[zWnumf].append(float(zWnf))
        p[zBaccuracyf].append(float(zBaf))
        p[zBnumf].append(float(zBnf))

for pab2 in sorted(pAB.keys()):
    pab=pab2

    p=pAB[pab]
    nf='nf'
    combined=zip(p[xf],p[yf],p[zWaccuracyf],p[zWnumf],p[zBaccuracyf],p[zBnumf],p[nf])

    s=sorted(combined,key=lambda t: t[1])
    s2=sorted(s,key=lambda t: t[0])
    ll=list(s2)

    #Now go through and average any multiple x,y values
    xymap=[(x_,y_) for (x_, y_, zWa_, zWn_, zBa_, zBn_,nf_) in ll ]

    cylist=sorted(list(set(xymap)),key=lambda t: t[1])
    cylist=sorted(list(cylist),key=lambda t: t[0])
    print(cylist)
    #donexy=set(xymap) #this is used to average multiple results
    done=[]

    x=[]
    y=[]
    zWa=[]
    zWn=[]
    zBa=[]
    zBn=[]
    n=[]

    for xy in xymap:
        if xy not in done:
            done.append(xy)
            indices = [ii for ii, xx in enumerate(xymap) if xx == xy]
            sum=0
            qty=len(indices)
            zWa_ =0
            zWn_ =0
            zBa_ =0
            zBn_ =0
            nf_=0
            for i in indices:
                nsamples=ll[i][6]
                nf_+= ll[i][6]
                zWa_+=ll[i][2]*nsamples
                zWn_ += ll[i][3]*nsamples
                zBa_ += ll[i][4]*nsamples
                zBn_ += ll[i][5]*nsamples
            zWa_/=nf_
            zWn_/=nf_
            zBa_/=nf_
            zBn_/=nf_

            x.append(xy[0])
            y.append(xy[1])
            zWa.append(zWa_)
            zWn.append(zWn_)
            zBa.append(zBa_)
            zBn.append(zBn_)
            n.append(nf_)

        pass

    zWaccuracy=zWa
    zWnum=zWn
    zBaccuracy=zBa
    zBnum=zBn
    ###This sorting is prob not neccesary
    combined2=zip(x,y,zWaccuracy,zWnum,zBaccuracy,zBnum,n)
    s3=sorted(combined2,key=lambda t: t[1])
    #print(s3)
    s4=sorted(s3,key=lambda t: t[0])
    ll2=list(s4)

    x,y,zWaccuracy,zWnum,zBaccuracy,zBnum,nf=zip(*ll2)

    elev = 30
    az = 117
    from matplotlib import cm
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import axes3d
    #############################################Wilson accuracy
    n_extrapolatedPts = 200
    #############################################BayesContour

    fig5 = plt.figure(figsize=plt.figaspect(0.5))
    xi,yi,zi=interpretXYZ(x,y,zBaccuracy,n_extrapolatedPts)
    ax2 = fig5.add_subplot(1, 1, 1)
    cs2 = ax2.contour(xi, yi, zi, 50, linewidths=1, cmap=cm.jet)
    ax2.invert_yaxis()
    ax2.set_xlabel(r"$\alpha$")
    ax2.set_ylabel(r'$\Delta$')
    #ax2.set_zlabel('Accuracy')
    ax2.set_title(f"Accuracy using Bayes-U Pab={pab}")
    levels = cs2.levels
    plt.clabel(cs2, levels[1::5],inline=10, fontsize=10)
    try:
        plt.colorbar(cs2, ax=ax2)
    except: #if there is no variation then color wont show
        pass
    fig5.savefig(f"failureTest/contour_Bayesp={pab}.png", format='png')
    if show:
        fig5.show()
        plt.show()
    ############################################### WilsonContour
    fig = plt.figure(figsize=plt.figaspect(0.5))
    xi,yi,zi=interpretXYZ(x,y,zWaccuracy,n_extrapolatedPts)
    ax = fig.add_subplot(1, 1, 1)
    cs1 = ax.contour(xi, yi, zi, 50, linewidths=1,cmap=cm.jet)
    ax.invert_yaxis()
    ax.set_xlabel(r"$\alpha$")
    ax.set_ylabel(r'$\Delta$')
    #ax.set_zlabel('Accuracy')
    ax.set_title(f"Accuracy using Wilson Pab={pab}")
    levels = cs1.levels
    plt.clabel(cs1, levels[1::5], inline=10, fontsize=10)
    try:
        plt.colorbar(cs1, ax=ax)
    except: #if there is no variation then color wont show
        pass
    fig.savefig(f"failureTest/contour_Wilsonp={pab}.png", format='png')
    if show:
        fig.show()
        plt.show()
    ############################################### Wilson
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
    fig.savefig(f"failureTest/cont3d_Wilsonp={pab}.png", format='png')
    if show:
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
    plt.colorbar(cs2, ax=ax2)
    fig2.savefig(f"failureTest/cont3d_Bayesp={pab}.png", format='png')

    if show:
        fig2.show()
        plt.show()
    ###############################################
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
    fig.savefig(f"failureTest/cont3d_Wilsonp={pab}.png", format='png')
    if show:
        fig.show()
        plt.show()


    #############################################Wilson nGames

    fig3 = plt.figure(figsize=plt.figaspect(0.5))
    xi, yi, zi = interpretXYZ(x, y, zWnum, n_extrapolatedPts)
    ax3 = fig3.add_subplot(1, 1, 1, projection='3d')
    cs3 = ax3.contourf(xi, yi, zi, 500, linewidths=1, cmap=cm.jet)
    ax3.invert_yaxis()
    ax3.set_xlabel('alpha')
    # ax.set_xlim(0, maxNgames)
    ax3.set_ylabel('delta')
    ax3.set_zlabel('Ngames to decision')
    ax3.view_init(elev=elev, azim=az)
    ax3.set_title(f"NGames to decision Wilson Pab={pab}")
    plt.colorbar(cs3, ax=ax3)
    fig3.savefig(f"failureTest/nGames_Wilsonp={pab}.png", format='png')
    if show:
        fig3.show()
        plt.show()

    #############################################Bayes nGames Contour

    fig7 = plt.figure(figsize=plt.figaspect(0.5))
    xi, yi, zi = interpretXYZ(x, y, zBnum, n_extrapolatedPts)
    ax4 = fig7.add_subplot(1, 1, 1)
    cs4 = ax4.contour(xi, yi, zi, 50, linewidths=1, cmap=cm.jet)
    ax4.invert_yaxis()
    ax4.set_xlabel('alpha')
    # ax.set_xlim(0, maxNgames)
    ax4.set_ylabel('delta')
    #ax4.set_zlabel('Ngames to decision')
    levels = cs1.levels
    plt.clabel(cs1, levels[1::5], inline=10, fontsize=10)
    ax4.set_title(f"NGames to decision Bayes-U Pab={pab}")
    plt.colorbar(cs4, ax=ax4)
    fig7.savefig(f"failureTest/contour_nGames_Bayesp={pab}.png", format='png')
    if show:
        fig7.show()
        plt.show()
    #############################################Bayes nGames

    fig4 = plt.figure(figsize=plt.figaspect(0.5))
    xi, yi, zi = interpretXYZ(x, y, zBnum, n_extrapolatedPts)
    ax4 = fig4.add_subplot(1, 1, 1, projection='3d')
    cs4 = ax4.contourf(xi, yi, zi, 500, linewidths=1, cmap=cm.jet)
    ax4.invert_yaxis()
    ax4.set_xlabel('alpha')
    # ax.set_xlim(0, maxNgames)
    ax4.set_ylabel('delta')
    ax4.set_zlabel('Ngames to decision')
    ax4.view_init(elev=elev, azim=az)
    ax4.set_title(f"NGames to decision Bayes-U Pab={pab}")
    plt.colorbar(cs4, ax=ax4)
    fig4.savefig(f"failureTest/nGames_Bayesp={pab}.png", format='png')
    if show:
        fig4.show()
        plt.show()