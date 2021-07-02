import matplotlib.pyplot as plt
import numpy as np

def init_cancerGDV(
    xd = 0.04, 
    xg = 0.9, 
    xv = 0.06,
    ba = 2.5,
    bv = 2,
    c = 1,
    n_neigh = 4,
    dt = 0.0001,
    iter = 500000,
    rb = 10**(-1.5),
    fb = 10**(-1.5),
    d = 0, timeup = None, timelow = None 
):
    """Function to plot static evolution of cancer game.

    Arguments
    ---------
        xd (float): subpopulation proportion of DEF tumor; 
            default 0.04

        xg (float): subpopulation proportion of GLY tumor; 
            default 0.9

        xv (float): subpopulation proportion of VOP tumor; 
            default 0.06

        ba (float): the benefit per unit of acidification; 
            default 2.5

        bv (float): the benefit from thge oxygen per unit of vascularization; 
            default 2

        c (float): the cost of production VEGF; default 1

        n_neigh (float): the number of GLY cells in the interaction group;
            default 4

        dt (float): time differentiation; 
            default 0.0001

        iter (int): tumors' evolutionary time dependency;
            default 500000
        
        rb (float): recovery barrier;
            default 10**(-1.5)
        
        fb (float): failure barrier;
            default 10**(-1.5)
        
        d (float): constrant medicine; 
            default 0
        
        timeup (int): upper time threshold of adding d;
            default None

        timelow (int): lower time threshold of adding d;
            default None
    Returns
    -------
        A matplotlib figure object containing the designated simplex.
    """
    # Evolution of subpopulation propotions
    xdpoints = [xd]
    xgpoints = [xg]
    xvpoints = [xv]
    ppoints = [xg]
    qpoints = [xv/(xv + xd)]

    succeed = rb
    fail = 1-fb

    for t in range(iter):
        # 2-D tranformation
        q = xv/(xv + xd)
        p = xg

        sum_p = 0
        for k in range(0, n_neigh):
            sum_p += p**k
        
        # Replicator dynamic in 2-D transformation
        q = q + q * (1 - q) * (bv/(n_neigh+1) * sum_p - c) * dt
        if d != 0 and timeup != None and timelow != None:
            if t >= timelow and t < timeup:    
                p = p + p * (1 - p) * (ba/(n_neigh+1) - (bv - c) * q - d) * dt
        else:
            p = p + p * (1 - p) * (ba/(n_neigh+1) - (bv - c) * q) * dt
        
        # Convert from 2-D to 3-D
        xd = (1 - q) * (1 - p)
        xg = p
        xv = (1 - p) * q

        ppoints.append(p)
        qpoints.append(q)

        xdpoints.append(xd)
        xgpoints.append(xg)
        xvpoints.append(xv)
        
        # Terminal condition
        if p < succeed:
            print("Therapy succeed")
            break
        elif p > fail:
            print("Therapy fail")
            break

    # Constructing plot for visualization of the CancerGDV
    fig, ax = plt.subplots(2, figsize=(18,15))
    
    # Visualization of 2-D system
    ax[0].plot(qpoints, ppoints)
    ax[0].axhline(succeed, color="r", linestyle='dashed', label="Succeed barrier")
    ax[0].axhline(fail, color="g", linestyle='dashed', label="Fail barrier")

    ax[0].set_xlim(0, 1)
    ax[0].set_ylim(0, 1)

    ax[0].set_title("2-D representation of cancer game", fontweight="bold", fontsize='x-large')
    ax[0].set_xlabel("p points", fontweight="bold", fontsize='x-large')
    ax[0].set_ylabel("q points", fontweight="bold", fontsize='x-large')
    ax[0].legend()

    # Visualization of 3-D system
    length = len(xgpoints)
    ax[1].plot(np.arange(0, length, 1), xgpoints, label="GLY")
    ax[1].plot(np.arange(0, length, 1), xdpoints, label="DEF")
    ax[1].plot(np.arange(0, length, 1), xvpoints, label="VOP")

    ax[1].set_ylim(0, 1)

    # Color the changing part
    if timelow != None and timeup != None:
        if timeup < t:
            ax[1].axvspan(timelow, timeup, facecolor="red", alpha=0.25)
        else:
            ax[1].axvspan(timelow, t, facecolor="red", alpha=0.25)

    ax[1].set_title("3-D representation of cancer game", fontweight="bold", fontsize='x-large')
    ax[1].set_xlabel("Time", fontweight="bold", fontsize='x-large')
    ax[1].set_ylabel("Subpopulation proportions", fontweight="bold", fontsize='x-large')
    ax[1].legend()

    return fig