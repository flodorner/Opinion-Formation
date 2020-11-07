from model import coevolution_model_general,holme,weighted_balance,weighted_balance_bots,H
from matplotlib import pyplot as plt
import numpy as np
import os
import pickle
from datetime import datetime

 

def experiment_loop(kwarg_dict,variying_kwarg,metrics,n=100,model_type=None,t_lim=99999,verbose=False):
    timestamp=datetime.now().strftime("%Y-%m-%d %H:%M")
    np.random.seed=0
    results = {key: [] for key in metrics.keys()}
    for v_kwarg in variying_kwarg[1]:
        if verbose:
            print(v_kwarg)
        kwarg_dict[variying_kwarg[0]]=v_kwarg
        subresults = {key: [] for key in metrics.keys()}
        for i in range(n):
            print('iteration {} of {}'.format(i,n))
            if model_type == "Holme":
                A = holme(**kwarg_dict)
            elif model_type == "Weighted Balance":
                A = weighted_balance(**kwarg_dict)
            elif model_type == "Weighted Balance Bots":
                A = weighted_balance_bots(**kwarg_dict)
            else:
                print("using general mode")
                A = coevolution_model_general(**kwarg_dict)
            done = False
            while done == False:
                A.step()
                if A.t%100==0: #Finding connected components is way more complex than the model dynamics. Only check for convergence every 100 steps.
                    done = A.convergence()
                if A.t>t_lim:
                    print("Model did not converge")
                    done = True
            for key in metrics.keys():
                subresults[key].append(metrics[key](A))
            #save subresults after every iteration
            
            #with open("./subresults/run{}.pickle".format(timestamp), "wb") as f:
                #pickle.dump(subresults, f)    

        for key in subresults.keys():
            results[key].append(subresults[key])
        
    results["variation"] = variying_kwarg
    A.draw_graph(path = image_folder+"graph")
    return results

def median_plus_percentile_plot(x,y,color="orange",percentiles=[10]):
    medians = [np.median([condition]) for condition in y]
    plt.plot(x,medians,color=color)
    top_old = medians
    bottom_old = medians
    for i in range(len(percentiles)):
        top = [np.percentile(condition,100-percentiles[i]) for condition in y]
        plt.plot(x, top, color=color,alpha=1/(2+i),label='_nolegend_')
        plt.fill_between(x,top_old,top,color=color,alpha=1/(2+i),label='_nolegend_')
        top_old = top
    for i in range(len(percentiles)):
        bottom = [np.percentile(condition, percentiles[i]) for condition in y]
        plt.plot(x, bottom, color=color, alpha=1 / (2 + i),label='_nolegend_')
        plt.fill_between(x, bottom_old, bottom,color=color,alpha=1/(2+i),label='_nolegend_')
        bottom_old = bottom


dir_path = os.path.dirname(os.path.realpath(__file__))
image_folder = "\\".join(dir_path.split("\\")[:-2]) + "\\doc\\latex\\images\\"

metrics = {
    "time_to_convergence": lambda x:x.t,
    "mean_size_connected_component": lambda x: np.mean([len(k) for k in x.connected_components()]),
    "sd_size_connected_component": lambda x: np.sqrt(np.var([len(k) for k in x.connected_components()])),
    "followers_per_opinion": lambda x: [np.sum(x.vertices==i) for i in range(x.n_opinions)]
}

#Strategy: start with small number of n to explore what kind of results you want to add.
#Properly label everything
#In the end, time the simple loop and rerun everything with as large n as possible.
import cProfile
#cProfile.run("output = experiment_loop(kw,loop,metrics=metrics,n=50)")

# @david code structure proposal: different experiments in different functions
def experiment_holme_N25():
    n_iterations = 10
    for n_vertices in [25]:
        for n_opinions in [5]:
            for phi in [0.05,0.5,0.95]:
                kw={"n_vertices":n_vertices, "n_opinions":n_opinions,"phi":phi}
                print(kw)
                loop = ("n_edges",np.arange(1,101*(n_vertices/25)**2,4*(n_vertices/25)**2,dtype=np.int))
                output = experiment_loop(kw,loop,metrics=metrics,n=n_iterations,model_type="Holme")
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
                plt.close()

    

def experiment_WB25():
    n_iterations = 10
    kw={}
    print(kw)
    loop = ("n_vertices",np.arange(2,25,4,dtype=np.int))
    model_type="Weighted Balance"
    output = experiment_loop(kw,loop,metrics=metrics,n=n_iterations,model_type=model_type)

    import cProfile
#    cProfile.run("output=experiment_loop(kw,loop,metrics=metrics,n=n_iterations,model_type=model_type)",sort="cumtime")

    median_plus_percentile_plot(output["variation"][1],output["sd_size_connected_component"])
    plt.title("Sd of community size")
    plt.xlabel("Number of Edges")
    plt.savefig(image_folder+"sd_25")
    plt.close()
    median_plus_percentile_plot(output["variation"][1],output["mean_size_connected_component"])
    plt.title("Mean of community size")
    plt.xlabel("Number of Edges")
    plt.savefig(image_folder+"mean_25")
    plt.close()
    median_plus_percentile_plot(output["variation"][1],output["time_to_convergence"])
    plt.title("Time steps to convergence")
    plt.xlabel("Number of Edges")
    plt.savefig(image_folder+"t_25")
    plt.close()


def bot_plots(recover=False, both_sides= False, neutral_bots=False, edges=None, fontsize = 20):
    name = "bots"+both_sides*"_double"+neutral_bots*"_neutral"+(edges!=None)*("_edges_"+str(edges))
    colors = [(1, 0, 0), (0.5, 0, 0.5), (0, 0, 1)]
    if not recover:
        metrics = {
            "time_to_convergence": lambda x: x.t,
            "maximal absolute opinion": lambda x:
            np.max(np.abs(np.mean(x.vertices[x.n_bots:-x.n_bots,:-1],axis=0))) if (x.both_sides and x.n_bots>0) else np.max(np.abs(np.mean(x.vertices[x.n_bots:,:-1],axis=0))),
            "H": lambda x: H(x.vertices[x.n_bots:-x.n_bots,:-1],x.d-1) if (x.both_sides and x.n_bots>0) else H(x.vertices[x.n_bots:,:-1],x.d-1)
        }
        if not both_sides:
            loop = ("n_bots", [0,2,6,10,20,30,40,50,100,150,200,250])
        else:
            loop = ("n_bots", [0, 1, 3, 5, 10, 15, 20, 25, 50, 75, 100, 125])
        output_1 = experiment_loop(
            {"n_vertices":500,"d": 3, "alpha": 0.4,"f":lambda x: np.sign(x)*np.abs(x)**(0.5),"n_edges":edges,
             "both_sides":both_sides,"neutral_bots":neutral_bots}, loop, metrics, n=10,
            model_type="Weighted Balance Bots",verbose=True)
        output_2 = experiment_loop(
            {"n_vertices": 500, "d": 3, "alpha": 0.4, "f": lambda x: np.sign(x) * np.abs(x),"n_edges":edges,
             "both_sides":both_sides,"neutral_bots":neutral_bots}, loop, metrics, n=10,
            model_type="Weighted Balance Bots", verbose=True)
        output_3 = experiment_loop(
            {"n_vertices": 500, "d": 3, "alpha": 0.4, "f": lambda x: np.sign(x) * np.abs(x) ** (2),"n_edges":edges,
             "both_sides":both_sides,"neutral_bots":neutral_bots}, loop, metrics, n=10,
            model_type="Weighted Balance Bots", verbose=True)
        with open(name, "wb") as fp:
            pickle.dump([output_1, output_2, output_3],fp)
    else:
        with open(name, "rb") as fp:
            output_1,output_2,output_3 = pickle.load(fp)


    median_plus_percentile_plot(output_1["variation"][1], output_1["time_to_convergence"],color=colors[0])
    median_plus_percentile_plot(output_2["variation"][1], output_2["time_to_convergence"],color=colors[1])
    median_plus_percentile_plot(output_3["variation"][1], output_3["time_to_convergence"],color=colors[2])

    plt.ylim(0,101001)
    plt.yticks(ticks=[0,20000,40000,60000,80000,100000],labels=["0","2e4","4e4","6e4","8e4","1e5"],fontsize=fontsize-4)
    plt.xticks(fontsize=fontsize-4)
    plt.xlabel("Number of bots" +both_sides*" per side",fontsize=fontsize)
    plt.ylabel("Steps until convergence",fontsize=fontsize)
    plt.legend(["0.5","0","-1"],title="Value of e",fontsize=fontsize,title_fontsize=fontsize)
    plt.savefig(image_folder+"t_conv_"+name)
    plt.close()

    median_plus_percentile_plot(output_1["variation"][1], output_1["maximal absolute opinion"],color=colors[0])
    median_plus_percentile_plot(output_2["variation"][1], output_2["maximal absolute opinion"],color=colors[1])
    median_plus_percentile_plot(output_3["variation"][1], output_3["maximal absolute opinion"],color=colors[2])
    plt.ylim(-0.01,1.01)
    plt.yticks(fontsize=fontsize-4)
    plt.xticks(fontsize=fontsize-4)
    plt.xlabel("Number of bots",fontsize=fontsize)
    plt.ylabel("Maximal absolute mean opinion",fontsize=fontsize)
    plt.legend(["0.5", "0",  "-1"], title="Value of e",fontsize=fontsize,title_fontsize=fontsize)
    plt.savefig(image_folder+"maxabsop_"+name)
    plt.close()

    median_plus_percentile_plot(output_1["variation"][1], output_1["H"],color=colors[0])
    median_plus_percentile_plot(output_2["variation"][1], output_2["H"],color=colors[1])
    median_plus_percentile_plot(output_3["variation"][1], output_3["H"],color=colors[2])
    plt.ylim(-0.01,1.01)
    plt.yticks(fontsize=fontsize-4)
    plt.xticks(fontsize=fontsize-4)
    plt.xlabel("Number of bots",fontsize=fontsize)
    plt.ylabel("Hyperpolarization H(O)",fontsize=fontsize)
    plt.legend(["0.5",  "0",  "-1"], title="Value of e",fontsize=fontsize,title_fontsize=fontsize)
    plt.savefig(image_folder+"H_"+name)
    plt.close()

