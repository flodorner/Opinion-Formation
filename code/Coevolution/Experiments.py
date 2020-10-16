from model import coevolution_model_general,coevolution_model_base
from matplotlib import pyplot as plt
import numpy as np
import os

def experiment_loop(kwarg_dict,variying_kwarg,metrics,n=100):
    np.random.seed=0
    results = {key: [] for key in metrics.keys()}
    for v_kwarg in variying_kwarg[1]:
        kwarg_dict[variying_kwarg[0]]=v_kwarg
        subresults = {key: [] for key in metrics.keys()}
        for i in range(n):
            A = coevolution_model_base(**kwarg_dict)
            done = False
            while done == False:
                A.step()
                if A.t%100==0: #Finding connected components is way more complex than the model dynamics. Only check for convergence every 100 steps.
                    done = A.convergence()
            for key in metrics.keys():
                subresults[key].append(metrics[key](A))
        for key in subresults.keys():
            results[key].append(subresults[key])
    results["variation"] = variying_kwarg
    return results

def median_plus_percentile_plot(x,y,color="orange",percentiles=[10]):
    medians = [np.median([condition]) for condition in y]
    plt.plot(x,medians,color=color)
    top_old = medians
    bottom_old = medians
    for i in range(len(percentiles)):
        top = [np.percentile(condition,100-percentiles[i]) for condition in y]
        plt.plot(x, top, color=color,alpha=1/(2+i))
        plt.fill_between(x,top_old,top,color=color,alpha=1/(2+i))
        top_old = top
    for i in range(len(percentiles)):
        bottom = [np.percentile(condition, percentiles[i]) for condition in y]
        plt.plot(x, bottom, color=color, alpha=1 / (2 + i))
        plt.fill_between(x, bottom_old, bottom,color=color,alpha=1/(2+i))
        bottom_old = bottom


dir_path = os.path.dirname(os.path.realpath(__file__))
image_folder = "\\".join(dir_path.split("\\")[:-2]) + "\\doc\\latex\\images\\"

metrics = {
    "time_to_convergence": lambda x:x.t,
    "mean_size_connected_component": lambda x: np.mean([len(k) for k in x.connected_components()]),
    "sd_size_connected_component": lambda x: np.sqrt(np.var([len(k) for k in x.connected_components()])),
    "followers_per_opinion": lambda x: [np.sum(x.vertices==i) for i in range(x.n_opinions)]}

#Strategy: start with small number of n to explore what kind of results you want to add.
#Properly label everything
#In the end, time the simple loop and rerun everything with as large n as possible.
import cProfile
#cProfile.run("output = experiment_loop(kw,loop,metrics=metrics,n=50)")

n_iterations = 10
"""for n_vertices in [25,50,100]:
    for n_opinions in [2,5,10]:
        for phi in [0.05,0.5,0.95]:
            kw={"n_vertices":n_vertices, "n_opinions":n_opinions,"phi":phi}
            print(kw)
            loop = ("n_edges",np.arange(1,101*(n_vertices/25)**2,4*(n_vertices/25)**2,dtype=np.int))

            output = experiment_loop(kw,loop,metrics=metrics,n=n_iterations)
            median_plus_percentile_plot(output["variation"][1],output["sd_size_connected_component"])
            plt.title("Sd of community size for N={},ϕ={},|O|={} and varying M".format(n_vertices,phi,n_opinions))
            plt.xlabel("Number of Edges")
            plt.savefig(image_folder+"sd_25_N{}_ϕ{}_O{}".format(n_vertices,int(phi*100),n_opinions))
            plt.close()
            median_plus_percentile_plot(output["variation"][1],output["mean_size_connected_component"])
            plt.title("Mean of community size for N={},ϕ={},|O|={} and varying M".format(n_vertices,phi,n_opinions))
            plt.xlabel("Number of Edges")
            plt.savefig(image_folder+"mean_25_N{}_ϕ{}_O{}".format(n_vertices,int(phi*100),n_opinions))
            plt.close()
            median_plus_percentile_plot(output["variation"][1],output["time_to_convergence"])
            plt.title("Time steps to convergence for N={},ϕ={},|O|={} and varying M".format(n_vertices,phi,n_opinions))
            plt.xlabel("Number of Edges")
            plt.savefig(image_folder+"t_25_N{}_ϕ{}_O{}".format(n_vertices,int(phi*100),n_opinions))
            plt.close()"""

for n_vertices in [25,50,100]:
    for n_opinions in [2,5,10]:
        for f_edges in [1,1.5,2]:
            kw={"n_vertices":n_vertices, "n_opinions":n_opinions,"n_edges":int(f_edges*n_vertices)}
            print(kw)
            loop = ("phi",np.arange(0.05,1,0.05))

            output = experiment_loop(kw,loop,metrics=metrics,n=n_iterations)
            median_plus_percentile_plot(output["variation"][1],output["sd_size_connected_component"])
            plt.title("Sd of community size for N={},M={}*N,|O|={} and varying ϕ".format(n_vertices,f_edges,n_opinions))
            plt.xlabel("ϕ")
            plt.savefig(image_folder+"sd_25_N{}_M{}_O{}".format(n_vertices,int(f_edges*n_vertices),n_opinions))
            plt.close()
            median_plus_percentile_plot(output["variation"][1],output["mean_size_connected_component"])
            plt.title("Mean of community size for N={},M={}*N,|O|={} and varying ϕ".format(n_vertices,f_edges,n_opinions))
            plt.xlabel("ϕ")
            plt.savefig(image_folder+"mean_25_N{}_M{}_O{}".format(n_vertices,int(f_edges*n_vertices),n_opinions))
            plt.close()
            median_plus_percentile_plot(output["variation"][1],output["time_to_convergence"])
            plt.title("Time steps to convergence for N={},M={}*N,|O|={} and varying ϕ".format(n_vertices,f_edges,n_opinions))
            plt.xlabel("ϕ")
            plt.savefig(image_folder+"t_25_N{}_M{}_O{}".format(n_vertices,int(f_edges*n_vertices),n_opinions))
            plt.close()