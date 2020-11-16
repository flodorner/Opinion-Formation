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

#HOLME:py - modeling the coevolution of opinions and networks
#Based on the paper "Nonequilibrium phase transition in the coevolution
#       of networks and opinions" by P. Holme & M.E.J. Newman 2006


if not os.path.isdir("./subresults"):
    os.mkdir("./subresults")


#default metrics
metrics = {
"time_to_convergence": lambda x:x.t,
"max_connected_components": lambda x: np.max([len(k) for k in x.connected_components()]),
}
#"sd_size_connected_component": lambda x: np.sqrt(np.var([len(k) for k in x.connected_components()])),
#"size_connected_component": lambda x: [len(k) for k in x.connected_components()],
#"followers_per_opinion": lambda x: [np.sum(x.vertices==i) for i in range(x.n_opinions)]

def holme_experiment_loop(kwarg_dict={"n_opinions":5},variying_kwarg=("phi",np.array([0.5,0.6])),metrics=metrics,n=10):
    timestamp=datetime.now().strftime("%Y-%m-%d %H:%M")
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
            with open("./subresults/run{} phi{} iter{}.pickle".format(timestamp,v_kwarg,i), "wb") as f:
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
    results["model"]={"vertices":A.vertices,"adjacency":A.adjacency}
    
    if len(variying_kwarg[1])==1:
        run_name="run{} iter{} opinions{} phi{}".format(timestamp,n,A.n_opinions,variying_kwarg[1][0])
    else:
        run_name="run{} iter{} opinions{} n_phi{}".format(timestamp,n,A.n_opinions,len(variying_kwarg[1]))

    results["run_name"]=run_name
    while path.exists("./"+run_name):
        run_name=run_name+"a"
    os.mkdir("./"+run_name)   
    with open("./{}/result.pickle".format(run_name), "wb") as f:
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
    phi_arr=np.array([0.45,0.458,0.46,0.465,0.47,1])
    kw={"n_opinions":3}
    results=holme_experiment_loop(kwarg_dict=kw,variying_kwarg=("phi",phi_arr),metrics=metrics,n=n_iterations)
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


def comu_size_run(phi, n_iterations=400):
    
    metrics = {
    "time_to_convergence": lambda x:x.t,
    "size_connected_component": lambda x: [len(k) for k in x.connected_components()],
    }
    phi_arr=np.array([phi])
    kw={"n_opinions":64}
    results=holme_experiment_loop(kwarg_dict=kw,variying_kwarg=("phi",phi_arr),metrics=metrics,n=n_iterations)
    print(results)
    
    return results



#results=max_S_testrun()
#plt.scatter(results["variation"],results["max_comm_avg"])

"""
if __name__ == "__main__":
    import sys
    if len(sys.argv) == 2: 
        comu_size_run(float(sys.argv[1]))
"""
#command to run on the server
# nohup python3 Holme.py > run3phi0.495.log &