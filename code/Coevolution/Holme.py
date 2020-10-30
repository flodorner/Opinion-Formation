from Experiments import experiment_loop 
from datetime import datetime
import os
from matplotlib import pyplot as plt
import numpy as np

n_iterations=10

n_opinion=10
#parameter in holme paper
k=4 #k=2M/N
gamma=10 #=n_vertices/n_opinion 

n_vertices = n_opinion*gamma
phi=0.459

kw={"n_vertices":n_vertices, "n_opinions":n_opinion,"phi":phi}
print(kw)
loop = ("n_edges",np.array([n_vertices*k/2],dtype=np.int))
metrics = {
"time_to_convergence": lambda x:x.t,
"size_connected_component": lambda x: [len(k) for k in x.connected_components()],
"followers_per_opinion": lambda x: [np.sum(x.vertices==i) for i in range(x.n_opinions)]
}
#"sd_size_connected_component": lambda x: np.sqrt(np.var([len(k) for k in x.connected_components()])),
#"followers_per_opinion": lambda x: [np.sum(x.vertices==i) for i in range(x.n_opinions)]
results = experiment_loop(kw,loop,metrics=metrics,n=n_iterations,model_type="Holme")
print("size_connected_component",results["size_connected_component"])
print("followers_per_opinion",results["followers_per_opinion"])
#caluclate the occurences of community sizes
#occ[3]=number of communities with size 3 in this run
#+1 because we include size 0 (to match with indexing),occ[0]=0
occ=np.zeros(n_vertices+1)

#[0] because this results has multiple arrays if loop/variyng_kwargs is used
for sizelist in results["size_connected_component"][0]:
    for l in sizelist:
        occ[l]+=1 #index is community size
distribution_commu_sizes=occ/sum(occ)

results["distribution_commu_sizes"]=distribution_commu_sizes


#sizes index    
s_index=np.array(range(len(occ)))

#divide results and sum over exponentially sized intervals
#so that the dots dont overlap in the logplot
n_intervals=100
#create logaritmic linspacing then 
intervals=np.array([np.int(np.ceil(np.exp(k))) for k in np.linspace(np.log(s_index[0]+0.0001),np.log(s_index[-1]),n_intervals+1)])
#linear version, equally spaced intervals
#intervals=np.array(np.linspace(occ_index[0],occ_index[-1],n_intervals+1))

#middle of the interval is the new corresponding size
new_s_index=(intervals[0:-1]+intervals[1:])/2
new_dist=[sum(occ[ intervals[k] : intervals[k+1]]) for k in range(n_intervals)]
#because python is non-inclusive for end indices, add the last one
new_dist[-1]+=distribution_commu_sizes[-1]

plt.scatter(new_s_index,new_dist,s=80, facecolors='none', edgecolors='r')
plt.yscale("log")
plt.xscale("log")
axes = plt.gca()
axes.set_ylim([0.0005,None])
axes.set_xlim([0.9,None])
plt.title("P(s) size distribution. ϕ={}".format(kw["phi"]))
plt.xlabel("s size of community")

timestamp=datetime.now().strftime("%Y-%m-%d %H%M")
run_name="run{}iter{}vertices{}phi{}".format(timestamp,n_iterations,n_vertices,phi)
os.mkdir("./"+run_name)
dir_path = os.path.dirname(os.path.realpath(__file__))

plt.savefig(run_name + "/size_distribution")

import pickle
with open("./{}/result.pickle".format(run_name), "wb") as f:
    pickle.dump(results, f)
    
