from model import holme2 
from datetime import datetime
import os
from matplotlib import pyplot as plt
import numpy as np
import cProfile, pstats
import pickle

if not os.path.isdir("./subresults"):
    os.mkdir("./subresults")

"""
n_iterations=1

n_opinions=30


phi=0.458

#kw={"n_vertices":n_vertices, "n_opinions":n_opinions,"phi":phi}
print(kw)
n_edges=np.array([n_vertices*k/2],dtype=np.int)
loop = ("n_edges",n_edges)
"""
metrics = {
"time_to_convergence": lambda x:x.t,
"max_connected_components": lambda x: np.max([len(k) for k in x.connected_components()]),
}
#"sd_size_connected_component": lambda x: np.sqrt(np.var([len(k) for k in x.connected_components()])),
#"size_connected_component": lambda x: [len(k) for k in x.connected_components()],
#"followers_per_opinion": lambda x: [np.sum(x.vertices==i) for i in range(x.n_opinions)]





#experiment_loop(kw,loop,metrics=metrics,n=n_iterations,model_type="Holme")
#print("size_connected_component",results["size_connected_component"])
#print("followers_per_opinion",results["followers_per_opinion"])
n_iterations=100
metrics = {
"time_to_convergence": lambda x:x.t,
"max_connected_components": lambda x: np.max([len(k) for k in x.connected_components()]),
}
phi_arr=np.array([0,0.2,0.3,0.4,0.42,0.43,0.44,0.445,0.45,0.454,0.456,0.458,0.46,0.465,0.47,0.48,0.5,0.6,0.8,1])
kw={"n_opinions":5}
def holme_experiment_loop(kwarg_dict={"n_opinions":5},variying_kwarg=("phi",np.array([0.5,0.6])),metrics=metrics,n=10):
    timestamp=datetime.now().strftime("%Y-%m-%d %H:%M")

    np.random.seed=0
    results = {key: [] for key in metrics.keys()}
    for v_kwarg in variying_kwarg[1]:
        kwarg_dict[variying_kwarg[0]]=v_kwarg
        subresults = {key: [] for key in metrics.keys()}
        for i in range(n):
            print('iteration {} of {}'.format(i,n))
            
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

    results["run_parameters"]={**kw,"n_vertices":A.n_vertices ,"n_edges":A.n_edges , "n_iterations":n}
    results["model"]={"vertices":A.vertices,"adjacency":A.adjacency}

    run_name="run{} iter{} opinions{} n_phi{}".format(timestamp,n,A.n_opinions,len(variying_kwarg[1]))
    os.mkdir("./"+run_name)   
    with open("./{}/result.pickle".format(run_name), "wb") as f:
        pickle.dump(results, f)


    return results


def criticalpoint_run():
    n_iterations=100
    metrics = {
    "time_to_convergence": lambda x:x.t,
    "max_connected_components": lambda x: np.max([len(k) for k in x.connected_components()]),
    }
    phi_arr=np.array([0,0.2,0.3,0.4,0.42,0.43,0.44,0.445,0.45,0.454,0.456,0.458,0.46,0.465,0.47,0.48,0.5,0.6,0.8,1])
    kw={"n_opinions":50}
    results=holme_experiment_loop(kwarg_dict=kw,variying_kwarg=("phi",phi_arr),metrics=metrics,n=n_iterations)
    return results
def criticalpoint_testrun():
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
    results = criticalpoint_testrun()
    pr.disable()
    pr.print_stats()

    pr.dump_stats('cprofile_data')
    ps = pstats.Stats('cprofile_data')
    ps.sort_stats(pstats.SortKey.CUMULATIVE)
    ps.print_stats()
    print(results)
profiling_run()

def plot_commu_sizes(n_vertices,phi,results, n_intervals=100):
    #caluclate the occurences of community sizes
    #occ[3]=number of communities with size 3 in this run
    #+1 because we include size 0 (to match with indexing),occ[0]=0
    occ=np.zeros(A.n_vertices+1)

    #[0] because this results has multiple arrays if loop/variyng_kwargs is used
    for sizelist in results["size_connected_component"][0]:
        for l in sizelist:
            occ[l]+=1 #index is community size
    distribution_commu_sizes=occ/sum(occ)

    results["distribution_commu_sizes"]=distribution_commu_sizes

    #sizes index    
    s_index=np.array(range(len(distribution_commu_sizes)))

    #divide results and sum over exponentially sized intervals
    #so that the dots dont overlap in the logplot
    
    #create logaritmic linspacing then 
    intervals=np.array([np.int(np.ceil(np.exp(k))) for k in np.linspace(np.log(s_index[0]+0.0001),np.log(s_index[-1]),n_intervals+1)])
    #linear version, equally spaced intervals
    #intervals=np.array(np.linspace(distribution_commu_sizes_index[0],distribution_commu_sizes_index[-1],n_intervals+1))

    #middle of the interval is the new corresponding size
    new_s_index=(intervals[0:-1]+intervals[1:])/2
    new_dist=[sum(distribution_commu_sizes[ intervals[k] : intervals[k+1]]) for k in range(n_intervals)]
    #because python is non-inclusive for end indices, add the last one
    new_dist[-1]+=distribution_commu_sizes[-1]

    plt.scatter(new_s_index,new_dist,s=80, facecolors='none', edgecolors='r')
    plt.yscale("log")
    plt.xscale("log")
    axes = plt.gca()
    axes.set_ylim([0.0005,None])
    axes.set_xlim([0.9,None])
    plt.title("P(s) size distribution. ϕ={}".format(phi))
    plt.xlabel("s size of community")
    plt.savefig(run_name + "/size_distribution")

    return results

#command to run on the server
# nohup python3 Holme.py > run3phi0.495.log &