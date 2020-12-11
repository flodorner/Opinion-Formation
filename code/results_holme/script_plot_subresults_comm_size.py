from os import listdir, mkdir, path
import re, pickle
import numpy as np
from matplotlib import pyplot as plt
import re
lsdr=listdir("results_comm_size")


for subf in lsdr:
    phi=re.search("phi([0-9.]+)",subf)[1]
    with open("results_comm_size/"+subf, "rb") as f:
        results=pickle.load(f)
    #caluclate the occurences of community sizes
    #occ[3]=number of communities with size 3 in this run
    #+1 because we include size 0 (to match with indexing),occ[0]=0


    run_name=subf.replace(".pickle","")

    allsizes=np.array(())
    for sizelist in results["size_connected_component"]:
        allsizes=np.concatenate((allsizes,sizelist))
    
    #logarithmic binning
    n_intervals=150
    n_vertices=640
    bins=np.logspace(np.log10(0.9),np.log10(n_vertices),n_intervals+1)
    histo=np.histogram(allsizes,bins=bins)

    #no binning
    histo2=np.histogram(allsizes,bins=640)
    
    dist=histo[0]/np.sum(histo[0])
    x=histo[1][0:-1]
   
    fig=plt.figure(figsize=(4.5,2.7))
    ax = fig.add_axes([0.15, 0.18, 0.84, 0.70])
    if phi=="0.458":
        #plt.scatter(x,dist,s=8,facecolors='r')
        
        #plot without binning because it caused jumps in the data
        plt.scatter(histo2[1][0:-1],histo2[0]/np.sum(histo2[0]),s=1, facecolors='none', edgecolors='r')


        #something like a line fitting but nothing rigorous
        xfilter=(x <70) & (x>8)
        yfit=dist[xfilter]+0.0001
        z=np.polyfit(np.log(x[xfilter]),np.log(yfit),1,w=np.sqrt(yfit))
        # Ax^b=y -> log(y)=b*log(x)+log(a)
        p=lambda x : z[1] * x**z[0]
        #plt.plot([8,80],[p(8),p(80)])
        plt.plot(x[xfilter],p(x[xfilter])*20)

    else:
        plt.scatter(x,dist,s=80, facecolors='none', edgecolors='r')
    
    plt.yscale("log")
    plt.xscale("log")
    axes = plt.gca()
    axes.set_ylim([0.00005,None])
    axes.set_xlim([0.9,650])
    
    #axes.set_xlim([3,50])
    plt.title("community size distribution. ϕ={}".format(phi))
    plt.xlabel("s size of community")
    plt.ylabel("P(s)")
    plt.savefig( "size_distribution"+phi+"s.png")
    plt.close()

    
    


    