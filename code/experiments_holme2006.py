from model import holme2 
from datetime import datetime
import os
from matplotlib import pyplot as plt
import numpy as np
import cProfile, pstats
import pickle
import sys
import os.path
from os import path
import matplotlib as mpl

#HOLME:py - modeling the coevolution of opinions and networks

### USAGE
# for TESTING use commu_size_testrun(phi=0.4) and max_s_testrun()
# 
# FULL SIMULATION
# for a proper run, 6-24h is needed. max_S_run() and commu_size_run() have the right parameters for that. 
# its best to use cloud computing, and call this script with the shell as a background process (see last lines)
# after each iteration, the subresults are saved in the folder subresults, in case of errors or crashes.
# results and subresults are saved with pickle, to read then see results_holme/pickleprinter.py
# needs to be executed on a linux machine (for compatible filenames/folder structures)

# RESULTS & FIGURES
# full simulations results can be found in results_holme, together with the 2 scripts for plots.
# the plots can be made from the pickle-saved results, because of the long simulation time

### DESCRIPTION
#Based on the paper "Nonequilibrium phase transition in the coevolution
#       of networks and opinions" by P. Holme & M.E.J. Newman 2006

# comu_size runs the model for a certain phi and calculates a histogram/distribution 
#   of the occurence of different community (component) sizes upon convergence
# max_S runs the model for a range of different values of phi, calculating the average 
#   maximum community size for each phi





#default metrics
metrics = {
"time_to_convergence": lambda x:x.t,
"max_connected_components": lambda x: np.max([len(k) for k in x.connected_components()]),
}
#"sd_size_connected_component": lambda x: np.sqrt(np.var([len(k) for k in x.connected_components()])),
#"size_connected_component": lambda x: [len(k) for k in x.connected_components()],
#"followers_per_opinion": lambda x: [np.sum(x.vertices==i) for i in range(x.n_opinions)]

def holme_experiment_loop(kwarg_dict={"n_opinions":5},variying_kwarg=("phi",np.array([0.5,0.6])),metrics=metrics,n=10, do_save_pickle=True):
    timestamp=datetime.now().strftime("%Y-%m-%d %H:%M")
    time_start=datetime.now()
    np.random.seed=0
    results = {key: [] for key in metrics.keys()}
    for v_kwarg in variying_kwarg[1]:
        kwarg_dict[variying_kwarg[0]]=v_kwarg
        subresults = {key: [] for key in metrics.keys()}
        for i in range(n):
            print('iteration {} of {}'.format(i,n))
            sys.stdout.flush()
            A = holme2(**kwarg_dict)
            done = False
            while done == False:
                A.step()
                 #Finding connected components is way more complex than the model dynamics. 
                 # Only check for convergence, if the opinions stopped changing

                if A.t%(A.n_vertices+3)==0:
                    if A.has_changed()== False:
                        done = A.convergence()
                        #print("has not changed step "+str(A.t))
            for key in metrics.keys():
                subresults[key].append(metrics[key](A))
            #save subresults after every iteration
            if do_save_pickle==True: 
                #if not os.path.isdir("./subresults"):
                    #os.mkdir("./subresults")
                with open("subresultsHolmeRun{} phi{} iter{}.pickle".format(timestamp,v_kwarg,i), "wb") as f:
                    pickle.dump(subresults, f)    

        for key in subresults.keys():
            results[key].append(subresults[key])
        
    results["variation"] = variying_kwarg
    ### Calculating additional metrics
    if "max_connected_components" in results.keys():
        results["max_comm_avg"] = np.average(results["max_connected_components"],1)
        results["max_comm_sd"] = np.std(results["max_connected_components"],1)

    results["run_time"]=str(datetime.now()-time_start)
    results["run_parameters"]={**kwarg_dict,"n_vertices":A.n_vertices ,"n_edges":A.n_edges , "n_iterations":n}
    results["model"]={"vertices":A.vertices,"graph":A.graph}
    
    if len(variying_kwarg[1])==1:
        run_name="holmeRun{} iter{} opinions{} phi{}".format(timestamp,n,A.n_opinions,variying_kwarg[1][0])
    else:
        run_name="holmeRun{} iter{} opinions{} n_phi{}".format(timestamp,n,A.n_opinions,len(variying_kwarg[1]))

    results["run_name"]=run_name
    if do_save_pickle==True:
        while path.exists("./"+run_name):
            run_name=run_name+"a"
        #os.mkdir("./"+run_name)   
        with open("results{}.pickle".format(run_name), "wb") as f:
            pickle.dump(results, f)


    return results


def max_S_run():
    n_iterations=100
    metrics = {
    "time_to_convergence": lambda x:x.t,
    "max_connected_components": lambda x: np.max([len(k) for k in x.connected_components()]),
    }
    phi_arr=np.array([0,0.2,0.3,0.4,0.42,0.43,0.44,0.45,0.454,0.456,0.458,0.46,0.465,0.47,0.48,0.5,0.6,0.8,1])
    kw={"n_opinions":50}
    results=holme_experiment_loop(kwarg_dict=kw,variying_kwarg=("phi",phi_arr),metrics=metrics,n=n_iterations)
    return results
def max_S_testrun():
    n_iterations=3
    metrics = {
    "time_to_convergence": lambda x:x.t,
    "max_connected_components": lambda x: np.max([len(k) for k in x.connected_components()]),
    }
    phi_arr=np.array([0.1,0.3,0.4,0.45,0.465,0.47,0.8,1])
    kw={"n_opinions":3}
    results=holme_experiment_loop(kwarg_dict=kw,variying_kwarg=("phi",phi_arr),metrics=metrics,n=n_iterations,do_save_pickle=True)
    plot_max_s(results)
    return results

def profiling_run():
    pr = cProfile.Profile()
    pr.enable()
    results = max_S_testrun()
    pr.disable()
    pr.print_stats()

    pr.dump_stats('cprofile_data')
    ps = pstats.Stats('cprofile_data')
    ps.sort_stats(pstats.SortKey.CUMULATIVE)
    ps.print_stats()
    print(results)


def commu_size_run(phi, n_iterations=400):
    
    metrics = {
    "time_to_convergence": lambda x:x.t,
    "size_connected_component": lambda x: [len(k) for k in x.connected_components()],
    }
    phi_arr=np.array([phi])
    kw={"n_opinions":64}
    results=holme_experiment_loop(kwarg_dict=kw,variying_kwarg=("phi",phi_arr),metrics=metrics,n=n_iterations)
    print(results)
    
    return results

def commu_size_testrun(phi, n_iterations=9):
    
    metrics = {
    "time_to_convergence": lambda x:x.t,
    "size_connected_component": lambda x: [len(k) for k in x.connected_components()],
    }
    phi_arr=np.array([phi])
    kw={"n_opinions":4}
    results=holme_experiment_loop(kwarg_dict=kw,variying_kwarg=("phi",phi_arr),metrics=metrics,n=n_iterations,do_save_pickle=False)
    print(results)
    plot_commu_size(results,results["run_parameters"]["n_vertices"],phi)
    
    return results
def plot_commu_size(results,n_vertices,phi):
    allsizes=np.array(())
    for sizelist in results["size_connected_component"][0]:
        allsizes=np.concatenate((allsizes,sizelist))    
    #logarithmic binning
    n_intervals=100
    
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

    plt.show()

def plot_max_s(results):
    x=results["variation"][1] #phi
    n=len(x)
    i=0
    y=np.zeros(n)
    sd=np.zeros(n)
    

    for subresult in results["max_connected_components"]:
        
            
        y[i]=np.mean(subresult)
        sd[i]=np.std(subresult)
        i=i+1

   
    #results={"x":x,"y":y,"sd":sd}



    mpl.rcParams['errorbar.capsize'] = 3

    fig=plt.figure(figsize=(5,3.4))
    ax = fig.add_axes([0.12, 0.15, 0.87, 0.74])
    plt.errorbar(x,y,yerr=sd,fmt="o")
    sorti=np.argsort(x)

    print(x,y,sd)

    plt.ylabel("S")
    plt.xlabel("ϕ")
    plt.title("Max community size phase transition")
    plt.plot([0.38,0.5],[48, 48], linewidth=4)
    plt.errorbar(x[sorti],y[sorti])
    plt.savefig("maxS")
    plt.show()

    fig=plt.figure(figsize=(4,3))
    ax = fig.add_axes([0.15, 0.15, 0.78, 0.7])
    plt.errorbar(x,y,yerr=sd,fmt="o")
    plt.xlim((0.38,0.5))
    plt.ylabel("S")
    plt.xlabel("ϕ")

    plt.plot([0.39,0.495],[48, 48], linewidth=4)
    plt.title("phase transition detail")
    plt.savefig("maxSdetail")
    plt.show()

#command to run on the server in the background
# nohup python3 Holme.py 0.495 > run3phi0.495.log &
"""
if __name__ == "__main__":
    import sys
    if len(sys.argv) == 2: 
        commu_size_run(float(sys.argv[1]))
"""


#### TESTING:

#max_S_testrun()
#commu_size_testrun(phi=0.9)







