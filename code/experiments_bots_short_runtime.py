from code.model import coevolution_model_general,holme,weighted_balance,weighted_balance_bots,H
from matplotlib import pyplot as plt
import numpy as np
import os
import pickle
from datetime import datetime

##### USAGE
# at the end of the file you find the figure numbers from the report next to the functions that generate them



## alternative parameters for testing/ shorter run time
# run time for all figures combined ~1h
n_bots2side= [0,2,6,10,14,20]
n_bots1side= [0, 1, 3, 5, 10, 14,20]
n_vertices=50
n_iterations=4
t_lim_default=39999
t_lim249=124999


"""
#parameters of original experiments, runtime multiple days (>48h) for all plots
n_bots2side= [0,2,6,10,20,30,40,50,100,150,200,250]
n_bots1side= [0, 1, 3, 5, 10, 15, 20, 25, 50, 75, 100, 125]
n_vertices=500
n_iterations=10
t_lim_default=99999
t_lim249=249999
"""
 

def experiment_loop(kwarg_dict,variying_kwarg,metrics,n=10,model_type=None,t_lim=t_lim_default,verbose=False):
    ''' runs `model_type` with options kwarg_dict for each variying_kwarg,  
                    each for n times then calculating metrics on model
        
        ARGUMENTS
        kwarg_dict: dict of keyword arguments that will be passed to the model class __init__ fun (check corresponding class in model.py)
        variying_kwarg: tuple with key and array of varying args e.g. ('phi', [0.1,0.5,0.7]), model is run for each one
        metrics: dict of functions called on the model object after each completed run, determines output in results
        n: int number of iterations for 
        model_type: Possible values "Holme", "Weighted Balance", "Weighted Balance General", "Weighted Balance Bot"


        RETURN
        result of metrics

    '''
    timestamp=datetime.now().strftime("%Y-%m-%d %H-%M")
    np.random.seed=0
    results = {key: [] for key in metrics.keys()}
    for v_kwarg in variying_kwarg[1]:
        if verbose:
            print(str(v_kwarg) + " from "+str(variying_kwarg[1]))
        kwarg_dict[variying_kwarg[0]]=v_kwarg
        subresults = {key: [] for key in metrics.keys()}
        for i in range(n):
            print('iteration {} of {}. '.format(i,n))
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
    # A.draw_graph(path = image_folder+"graph")
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


#dir_path = os.path.dirname(os.path.realpath(__file__))
#image_folder = "\\".join(dir_path.split("\\")[:-2]) + "\\doc\\latex\\images\\"
image_folder=""

metrics = {
    "time_to_convergence": lambda x:x.t,
    "mean_size_connected_component": lambda x: np.mean([len(k) for k in x.connected_components()]),
    "sd_size_connected_component": lambda x: np.sqrt(np.var([len(k) for k in x.connected_components()])),
    "followers_per_opinion": lambda x: [np.sum(x.vertices==i) for i in range(x.n_opinions)]
}

##### BOT PLOT FUNCTIONS
## Long execution times >1h
def bot_plots(recover=False, both_sides= False, neutral_bots=False, edges=None, fontsize = 20,t_lim=None,report_fig=""):
    ''' calculates and plots: (time)steps to convergence, mean opinion and hyperpolarization
            x-axis n_bots, for evaluative extremeness e= 0.5, 0 and -1 
            '''
    
    name = "bots"+both_sides*"_double"+neutral_bots*"_neutral"+(edges!=None)*("_edges_"+str(edges))+(t_lim!=None)*("_tlim_"+str(t_lim))+"fig_"+report_fig
    if t_lim == None:
        t_lim = t_lim_default
    colors = [(1, 0, 0), (0.5, 0, 0.5), (0, 0, 1)]
    if not recover:
        metrics = {
            "time_to_convergence": lambda x: x.t,
            "maximal absolute opinion": lambda x:
            np.max(np.abs(np.mean(x.vertices[x.n_bots:-x.n_bots,:-1],axis=0))) if (x.both_sides and x.n_bots>0) else np.max(np.abs(np.mean(x.vertices[x.n_bots:,:-1],axis=0))),
            "H": lambda x: H(x.vertices[x.n_bots:-x.n_bots,:-1],x.d-1) if (x.both_sides and x.n_bots>0) else H(x.vertices[x.n_bots:,:-1],x.d-1)
        }
        if not both_sides:
            loop = ("n_bots", n_bots2side)
        else:
            loop = ("n_bots", n_bots1side)
        output_1 = experiment_loop(
            {"n_vertices":n_vertices,"d": 3, "alpha": 0.4,"f":lambda x: np.sign(x)*np.abs(x)**(0.5),"n_edges":edges,
             "both_sides":both_sides,"neutral_bots":neutral_bots}, loop, metrics, n=n_iterations,
            model_type="Weighted Balance Bots",verbose=True,t_lim=t_lim)
        output_2 = experiment_loop(
            {"n_vertices":n_vertices, "d": 3, "alpha": 0.4, "f": lambda x: np.sign(x) * np.abs(x),"n_edges":edges,
             "both_sides":both_sides,"neutral_bots":neutral_bots}, loop, metrics, n=n_iterations,
            model_type="Weighted Balance Bots", verbose=True,t_lim=t_lim)
        output_3 = experiment_loop(
            {"n_vertices":n_vertices, "d": 3, "alpha": 0.4, "f": lambda x: np.sign(x) * np.abs(x) ** (2),"n_edges":edges,
             "both_sides":both_sides,"neutral_bots":neutral_bots}, loop, metrics, n=n_iterations,
            model_type="Weighted Balance Bots", verbose=True,t_lim=t_lim)
        with open(name+".pickle", "wb") as fp:
            pickle.dump([output_1, output_2, output_3],fp)
    else:
        with open(name+".pickle", "rb") as fp:
            output_1,output_2,output_3 = pickle.load(fp)


    median_plus_percentile_plot(output_1["variation"][1], output_1["time_to_convergence"],color=colors[0])
    median_plus_percentile_plot(output_2["variation"][1], output_2["time_to_convergence"],color=colors[1])
    median_plus_percentile_plot(output_3["variation"][1], output_3["time_to_convergence"],color=colors[2])

    plt.ylim(0,t_lim*101001/99999)
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

def bot_plots_altf(recover=False, both_sides= False, neutral_bots=False, edges=None, fontsize = 20,t_lim=None,report_fig=""):
    '''bots in models with alternative functions for evaluating the attitude: f = x +- 0.1, max or min(x,0)
    '''
    name = "bots_altf"+both_sides*"_double"+neutral_bots*"_neutral"+(edges!=None)*("_edges_"+str(edges))+(t_lim!=None)*("_tlim_"+str(t_lim))+"fig_"+report_fig
    if t_lim == None:
        t_lim = t_lim_default
    colors = [(1, 0, 0), (0,1,1),(1,1,0),(0,0,1)]
    if not recover:
        metrics = {
            "time_to_convergence": lambda x: x.t,
            "maximal absolute opinion": lambda x:
            np.max(np.abs(np.mean(x.vertices[x.n_bots:-x.n_bots,:-1],axis=0))) if (x.both_sides and x.n_bots>0) else np.max(np.abs(np.mean(x.vertices[x.n_bots:,:-1],axis=0))),
            "H": lambda x: H(x.vertices[x.n_bots:-x.n_bots,:-1],x.d-1) if (x.both_sides and x.n_bots>0) else H(x.vertices[x.n_bots:,:-1],x.d-1)
        }
        if not both_sides:
            loop = ("n_bots", n_bots2side)
        else:
            loop = ("n_bots", n_bots1side)
        output_1 = experiment_loop(
            {"n_vertices":n_vertices,"d": 3, "alpha": 0.4,"f":lambda x: x+0.1,"n_edges":edges,
             "both_sides":both_sides,"neutral_bots":neutral_bots}, loop, metrics, n=n_iterations,
            model_type="Weighted Balance Bots",verbose=True,t_lim=t_lim)
        output_2 = experiment_loop(
            {"n_vertices":n_vertices,"d": 3, "alpha": 0.4,"f":lambda x: x-0.1,"n_edges":edges,
             "both_sides":both_sides,"neutral_bots":neutral_bots}, loop, metrics, n=n_iterations,
            model_type="Weighted Balance Bots",verbose=True,t_lim=t_lim)
        output_3 = experiment_loop(
            {"n_vertices":n_vertices,"d": 3, "alpha": 0.4,"f":lambda x: max(0,x),"n_edges":edges,
             "both_sides":both_sides,"neutral_bots":neutral_bots}, loop, metrics, n=n_iterations,
            model_type="Weighted Balance Bots",verbose=True,t_lim=t_lim)
        output_4 = experiment_loop(
            {"n_vertices":n_vertices,"d": 3, "alpha": 0.4,"f":lambda x: min(0,x),"n_edges":edges,
             "both_sides":both_sides,"neutral_bots":neutral_bots}, loop, metrics, n=n_iterations,
            model_type="Weighted Balance Bots",verbose=True,t_lim=t_lim)



        with open(name+".pickle", "wb") as fp:
            pickle.dump([output_1, output_2, output_3,output_4],fp)
    else:
        with open(name+".pickle", "rb") as fp:
            output_1,output_2,output_3,output_4 = pickle.load(fp)


    median_plus_percentile_plot(output_1["variation"][1], output_1["time_to_convergence"],color=colors[0])
    median_plus_percentile_plot(output_2["variation"][1], output_2["time_to_convergence"],color=colors[1])
    median_plus_percentile_plot(output_3["variation"][1], output_3["time_to_convergence"],color=colors[2])
    median_plus_percentile_plot(output_4["variation"][1], output_4["time_to_convergence"], color=colors[3])

    plt.ylim(0,t_lim*101001/99999)
    plt.yticks(ticks=[0,20000,40000,60000,80000,100000],labels=["0","2e4","4e4","6e4","8e4","1e5"],fontsize=fontsize-4)
    plt.xticks(fontsize=fontsize-4)
    plt.xlabel("Number of bots" +both_sides*" per side",fontsize=fontsize)
    plt.ylabel("Steps until convergence",fontsize=fontsize)
    plt.legend(["x+0.1","x-0.1","max(x,0)","min(x,0)"],title="f(x)",fontsize=fontsize-4,title_fontsize=fontsize-4)
    plt.savefig(image_folder+"t_conv_"+name)
    plt.close()

    median_plus_percentile_plot(output_1["variation"][1], output_1["maximal absolute opinion"],color=colors[0])
    median_plus_percentile_plot(output_2["variation"][1], output_2["maximal absolute opinion"],color=colors[1])
    median_plus_percentile_plot(output_3["variation"][1], output_3["maximal absolute opinion"],color=colors[2])
    median_plus_percentile_plot(output_4["variation"][1], output_4["maximal absolute opinion"], color=colors[3])
    plt.ylim(-0.01,1.01)
    plt.yticks(fontsize=fontsize-4)
    plt.xticks(fontsize=fontsize-4)
    plt.xlabel("Number of bots",fontsize=fontsize)
    plt.ylabel("Maximal absolute mean opinion",fontsize=fontsize)
    plt.legend(["x+0.1","x-0.1","max(x,0)","min(x,0)"],title="f(x)",fontsize=fontsize-4,title_fontsize=fontsize-4)
    plt.savefig(image_folder+"maxabsop_"+name)
    plt.close()

    median_plus_percentile_plot(output_1["variation"][1], output_1["H"],color=colors[0])
    median_plus_percentile_plot(output_2["variation"][1], output_2["H"],color=colors[1])
    median_plus_percentile_plot(output_3["variation"][1], output_3["H"],color=colors[2])
    median_plus_percentile_plot(output_4["variation"][1], output_4["H"], color=colors[3])
    plt.ylim(-0.01,1.01)
    plt.yticks(fontsize=fontsize-4)
    plt.xticks(fontsize=fontsize-4)
    plt.xlabel("Number of bots",fontsize=fontsize)
    plt.ylabel("Hyperpolarization H(O)",fontsize=fontsize)
    plt.legend(["x+0.1","x-0.1","max(x,0)","min(x,0)"],title="f(x)",fontsize=fontsize-4,title_fontsize=fontsize-4)
    plt.savefig(image_folder+"H_"+name)
    plt.close()

def bot_plots_ba(recover=False, both_sides= False, neutral_bots=False, edges=None, fontsize = 20,bot_positions = None,t_lim=None,report_fig=""):
    ''' bots in static networks with barabasi-albert structure'''
    name = "bots_ba_"+(bot_positions!=None)*str(bot_positions)+both_sides*"_double"+neutral_bots*"_neutral"+(edges!=None)*("_edges_"+str(edges))+(t_lim!=None)*("_tlim_"+str(t_lim))+"fig_"+report_fig
    if t_lim == None:
        t_lim = t_lim_default
    colors = [(1, 0, 0), (0.5, 0, 0.5), (0, 0, 1)]
    if not recover:
        metrics = {
            "time_to_convergence": lambda x: x.t,
            "maximal absolute opinion": lambda x:
            np.max(np.abs(np.mean(x.vertices[np.logical_not(x.bot_indices)][:,:-1],axis=0))),
            "H": lambda x: H(x.vertices[np.logical_not(x.bot_indices)][:,:-1],x.d-1)
        }
        if not both_sides:
            loop = ("n_bots", n_bots2side)
        else:
            loop = ("n_bots", n_bots1side)
        output_1 = experiment_loop(
            {"n_vertices":n_vertices,"d": 3, "alpha": 0.4,"f":lambda x: np.sign(x)*np.abs(x)**(0.5),"n_edges":edges,
             "both_sides":both_sides,"neutral_bots":neutral_bots,"initial_graph":"barabasi_albert","bot_positions":bot_positions}, loop, metrics, n=n_iterations,
            model_type="Weighted Balance Bots",verbose=True,t_lim=t_lim)
        output_2 = experiment_loop(
            {"n_vertices":n_vertices, "d": 3, "alpha": 0.4, "f": lambda x: np.sign(x) * np.abs(x),"n_edges":edges,
             "both_sides":both_sides,"neutral_bots":neutral_bots,"initial_graph":"barabasi_albert","bot_positions":bot_positions}, loop, metrics, n=n_iterations,
            model_type="Weighted Balance Bots", verbose=True,t_lim=t_lim)
        output_3 = experiment_loop(
            {"n_vertices":n_vertices, "d": 3, "alpha": 0.4, "f": lambda x: np.sign(x) * np.abs(x) ** (2),"n_edges":edges,
             "both_sides":both_sides,"neutral_bots":neutral_bots,"initial_graph":"barabasi_albert","bot_positions":bot_positions}, loop, metrics, n=n_iterations,
            model_type="Weighted Balance Bots", verbose=True,t_lim=t_lim)
        with open(name+".pickle", "wb") as fp:
            pickle.dump([output_1, output_2, output_3],fp)
    else:
        with open(name+".pickle", "rb") as fp:
            output_1,output_2,output_3 = pickle.load(fp)


    median_plus_percentile_plot(output_1["variation"][1], output_1["time_to_convergence"],color=colors[0])
    median_plus_percentile_plot(output_2["variation"][1], output_2["time_to_convergence"],color=colors[1])
    median_plus_percentile_plot(output_3["variation"][1], output_3["time_to_convergence"],color=colors[2])

    plt.ylim(0,t_lim*101001/99999)
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

def bot_plots_altf_ba(recover=False, both_sides= False, neutral_bots=False, edges=None, fontsize = 20,t_lim=None,bot_positions = None,report_fig=""):
    name = "bots_altf_ba_"+(bot_positions!=None)*str(bot_positions)+both_sides*"_double"+neutral_bots*"_neutral"+(edges!=None)*("_edges_"+str(edges))+(t_lim!=None)*("_tlim_"+str(t_lim))+"fig_"+report_fig
    if t_lim == None:
        t_lim = t_lim_default
    colors = [(1, 0, 0), (0,1,1),(1,1,0),(0,0,1)]
    if not recover:
        metrics = {
            "time_to_convergence": lambda x: x.t,
            "maximal absolute opinion": lambda x:
            np.max(np.abs(np.mean(x.vertices[np.logical_not(x.bot_indices)][:, :-1], axis=0))),
             "H": lambda x: H(x.vertices[np.logical_not(x.bot_indices)][:,:-1],x.d-1)
        }
        if not both_sides:
            loop = ("n_bots", n_bots2side)
        else:
            loop = ("n_bots", n_bots1side)
        output_1 = experiment_loop(
            {"n_vertices":n_vertices,"d": 3, "alpha": 0.4,"f":lambda x: x+0.1,"n_edges":edges,
             "both_sides":both_sides,"neutral_bots":neutral_bots,"initial_graph":"barabasi_albert","bot_positions":bot_positions}, loop, metrics, n=n_iterations,
            model_type="Weighted Balance Bots",verbose=True,t_lim=t_lim)
        output_2 = experiment_loop(
            {"n_vertices":n_vertices,"d": 3, "alpha": 0.4,"f":lambda x: x-0.1,"n_edges":edges,
             "both_sides":both_sides,"neutral_bots":neutral_bots,"initial_graph":"barabasi_albert","bot_positions":bot_positions}, loop, metrics, n=n_iterations,
            model_type="Weighted Balance Bots",verbose=True,t_lim=t_lim)
        output_3 = experiment_loop(
            {"n_vertices":n_vertices,"d": 3, "alpha": 0.4,"f":lambda x: max(0,x),"n_edges":edges,
             "both_sides":both_sides,"neutral_bots":neutral_bots,"initial_graph":"barabasi_albert","bot_positions":bot_positions}, loop, metrics, n=n_iterations,
            model_type="Weighted Balance Bots",verbose=True,t_lim=t_lim)
        output_4 = experiment_loop(
            {"n_vertices":n_vertices,"d": 3, "alpha": 0.4,"f":lambda x: min(0,x),"n_edges":edges,
             "both_sides":both_sides,"neutral_bots":neutral_bots,"initial_graph":"barabasi_albert","bot_positions":bot_positions}, loop, metrics, n=n_iterations,
            model_type="Weighted Balance Bots",verbose=True,t_lim=t_lim)



        with open(name+".pickle", "wb") as fp:
            pickle.dump([output_1, output_2, output_3,output_4],fp)
    else:
        with open(name+".pickle", "rb") as fp:
            output_1,output_2,output_3,output_4 = pickle.load(fp)


    median_plus_percentile_plot(output_1["variation"][1], output_1["time_to_convergence"],color=colors[0])
    median_plus_percentile_plot(output_2["variation"][1], output_2["time_to_convergence"],color=colors[1])
    median_plus_percentile_plot(output_3["variation"][1], output_3["time_to_convergence"],color=colors[2])
    median_plus_percentile_plot(output_4["variation"][1], output_4["time_to_convergence"], color=colors[3])

    plt.ylim(0,t_lim*101001/99999)
    plt.yticks(ticks=[0,20000,40000,60000,80000,100000],labels=["0","2e4","4e4","6e4","8e4","1e5"],fontsize=fontsize-4)
    plt.xticks(fontsize=fontsize-4)
    plt.xlabel("Number of bots" +both_sides*" per side",fontsize=fontsize)
    plt.ylabel("Steps until convergence",fontsize=fontsize)
    plt.legend(["x+0.1","x-0.1","max(x,0)","min(x,0)"],title="f(x)",fontsize=fontsize-4,title_fontsize=fontsize-4)
    plt.savefig(image_folder+"t_conv_"+name)
    plt.close()

    median_plus_percentile_plot(output_1["variation"][1], output_1["maximal absolute opinion"],color=colors[0])
    median_plus_percentile_plot(output_2["variation"][1], output_2["maximal absolute opinion"],color=colors[1])
    median_plus_percentile_plot(output_3["variation"][1], output_3["maximal absolute opinion"],color=colors[2])
    median_plus_percentile_plot(output_4["variation"][1], output_4["maximal absolute opinion"], color=colors[3])
    plt.ylim(-0.01,1.01)
    plt.yticks(fontsize=fontsize-4)
    plt.xticks(fontsize=fontsize-4)
    plt.xlabel("Number of bots",fontsize=fontsize)
    plt.ylabel("Maximal absolute mean opinion",fontsize=fontsize)
    plt.legend(["x+0.1","x-0.1","max(x,0)","min(x,0)"],title="f(x)",fontsize=fontsize-4,title_fontsize=fontsize-4)
    plt.savefig(image_folder+"maxabsop_"+name)
    plt.close()

    median_plus_percentile_plot(output_1["variation"][1], output_1["H"],color=colors[0])
    median_plus_percentile_plot(output_2["variation"][1], output_2["H"],color=colors[1])
    median_plus_percentile_plot(output_3["variation"][1], output_3["H"],color=colors[2])
    median_plus_percentile_plot(output_4["variation"][1], output_4["H"], color=colors[3])
    plt.ylim(-0.01,1.01)
    plt.yticks(fontsize=fontsize-4)
    plt.xticks(fontsize=fontsize-4)
    plt.xlabel("Number of bots",fontsize=fontsize)
    plt.ylabel("Hyperpolarization H(O)",fontsize=fontsize)
    plt.legend(["x+0.1","x-0.1","max(x,0)","min(x,0)"],title="f(x)",fontsize=fontsize-4,title_fontsize=fontsize-4)
    plt.savefig(image_folder+"H_"+name)
    plt.close()


def bot_plots_ba_dynamic(recover=False, both_sides= False, neutral_bots=False, edges=None, fontsize = 20,bot_positions = None,t_lim=None,epsilon=1,seeking_bots=False,report_fig=""):
    name = "bots_ba_dyn"+seeking_bots*"_seeking_"+(bot_positions!=None)*str(bot_positions)+both_sides*"_double"+neutral_bots*"_neutral"+(edges!=None)*("_edges_"+str(edges))+"_epsilon_"+str(int(epsilon*10))+(t_lim!=None)*("_tlim_"+str(t_lim))+"fig_"+report_fig
    if t_lim == None:
        t_lim = t_lim_default
    colors = [(1, 0, 0), (0.5, 0, 0.5), (0, 0, 1)]
    if not recover:
        metrics = {
            "time_to_convergence": lambda x: x.t,
            "maximal absolute opinion": lambda x:
            np.max(np.abs(np.mean(x.vertices[np.logical_not(x.bot_indices)][:,:-1],axis=0))),
            "H": lambda x: H(x.vertices[np.logical_not(x.bot_indices)][:,:-1],x.d-1)
        }
        if not both_sides:
            loop = ("n_bots", n_bots2side)
        else:
            loop = ("n_bots", n_bots1side)
        output_1 = experiment_loop(
            {"n_vertices":n_vertices,"d": 3, "alpha": 0.4,"f":lambda x: np.sign(x)*np.abs(x)**(0.5),"n_edges":edges,
             "both_sides":both_sides,"neutral_bots":neutral_bots,"initial_graph":"barabasi_albert","bot_positions":bot_positions,"epsilon":epsilon,"phi":0.5,"seeking_bots":seeking_bots}, loop, metrics, n=n_iterations,
            model_type="Weighted Balance Bots",verbose=True,t_lim=t_lim)
        output_2 = experiment_loop(
            {"n_vertices":n_vertices, "d": 3, "alpha": 0.4, "f": lambda x: np.sign(x) * np.abs(x),"n_edges":edges,
             "both_sides":both_sides,"neutral_bots":neutral_bots,"initial_graph":"barabasi_albert","bot_positions":bot_positions,"epsilon":epsilon,"phi":0.5,"seeking_bots":seeking_bots}, loop, metrics, n=n_iterations,
            model_type="Weighted Balance Bots", verbose=True,t_lim=t_lim)
        output_3 = experiment_loop(
            {"n_vertices":n_vertices, "d": 3, "alpha": 0.4, "f": lambda x: np.sign(x) * np.abs(x) ** (2),"n_edges":edges,
             "both_sides":both_sides,"neutral_bots":neutral_bots,"initial_graph":"barabasi_albert","bot_positions":bot_positions,"epsilon":epsilon,"phi":0.5,"seeking_bots":seeking_bots}, loop, metrics, n=n_iterations,
            model_type="Weighted Balance Bots", verbose=True,t_lim=t_lim)
        with open(name+".pickle", "wb") as fp:
            pickle.dump([output_1, output_2, output_3],fp)
    else:
        with open(name+".pickle", "rb") as fp:
            output_1,output_2,output_3 = pickle.load(fp)


    median_plus_percentile_plot(output_1["variation"][1], output_1["time_to_convergence"],color=colors[0])
    median_plus_percentile_plot(output_2["variation"][1], output_2["time_to_convergence"],color=colors[1])
    median_plus_percentile_plot(output_3["variation"][1], output_3["time_to_convergence"],color=colors[2])

    plt.ylim(0,t_lim*101001/99999)
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

def bot_plots_shifted(recover=False, both_sides= False, neutral_bots=False, edges=None, fontsize = 20,t_lim=None,initial_opinion_range=[0,1],report_fig=""):
    '''effect on hyperpolarization and mean opinion with bots when constraining initial_opinion_range (and initial mean !=0)'''
    name = "bots"+both_sides*"_double"+neutral_bots*"_neutral"+(edges!=None)*("_edges_"+str(edges))+(t_lim!=None)*("_tlim_"+str(t_lim))+"shifted_"+str(int(100*initial_opinion_range[0]))+"_"+str(int(100*initial_opinion_range[1]))+"fig_"+report_fig
    if t_lim == None:
        t_lim = t_lim_default
    colors = [(1, 0, 0), (0.5, 0, 0.5), (0, 0, 1)]
    if not recover:
        metrics = {
            "time_to_convergence": lambda x: x.t,
            "maximal absolute opinion": lambda x:
            np.max(np.abs(np.mean(x.vertices[x.n_bots:-x.n_bots,:-1],axis=0))) if (x.both_sides and x.n_bots>0) else np.max(np.abs(np.mean(x.vertices[x.n_bots:,:-1],axis=0))),
            "H": lambda x: H(x.vertices[x.n_bots:-x.n_bots,:-1],x.d-1) if (x.both_sides and x.n_bots>0) else H(x.vertices[x.n_bots:,:-1],x.d-1)
        }
        if not both_sides:
            loop = ("n_bots", n_bots2side)
        else:
            loop = ("n_bots", n_bots1side)
        output_1 = experiment_loop(
            {"n_vertices":n_vertices,"d": 3, "alpha": 0.4,"f":lambda x: np.sign(x)*np.abs(x)**(0.5),"n_edges":edges,
             "both_sides":both_sides,"neutral_bots":neutral_bots,"initial_opinion_range":initial_opinion_range}, loop, metrics, n=n_iterations,
            model_type="Weighted Balance Bots",verbose=True,t_lim=t_lim)
        output_2 = experiment_loop(
            {"n_vertices":n_vertices, "d": 3, "alpha": 0.4, "f": lambda x: np.sign(x) * np.abs(x),"n_edges":edges,
             "both_sides":both_sides,"neutral_bots":neutral_bots,"initial_opinion_range":initial_opinion_range}, loop, metrics, n=n_iterations,
            model_type="Weighted Balance Bots", verbose=True,t_lim=t_lim)
        output_3 = experiment_loop(
            {"n_vertices":n_vertices, "d": 3, "alpha": 0.4, "f": lambda x: np.sign(x) * np.abs(x) ** (2),"n_edges":edges,
             "both_sides":both_sides,"neutral_bots":neutral_bots,"initial_opinion_range":initial_opinion_range}, loop, metrics, n=n_iterations,
            model_type="Weighted Balance Bots", verbose=True,t_lim=t_lim)
        with open(name+".pickle", "wb") as fp:
            pickle.dump([output_1, output_2, output_3],fp)
    else:
        with open(name+".pickle", "rb") as fp:
            output_1,output_2,output_3 = pickle.load(fp)


    median_plus_percentile_plot(output_1["variation"][1], output_1["time_to_convergence"],color=colors[0])
    median_plus_percentile_plot(output_2["variation"][1], output_2["time_to_convergence"],color=colors[1])
    median_plus_percentile_plot(output_3["variation"][1], output_3["time_to_convergence"],color=colors[2])

    plt.ylim(0,t_lim*101001/99999)
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


#### GENERATION OF BOT PLOTS FOR REPORT 
## the comment after each line is the figure number of the paper
def plot_99():
    bot_plots(fontsize=18,report_fig="14a,30a") #14a,30a
    bot_plots(fontsize=18,both_sides=True,report_fig="14b,30b") #14b,30b
    bot_plots(edges=n_vertices-1,fontsize=18,report_fig="15a,30c") #15a,30c
    bot_plots_ba(edges=n_vertices-1,fontsize=18,report_fig="15b,30d") #15b,30d
    bot_plots_ba(edges=n_vertices-1,fontsize=18,bot_positions="top",report_fig="16a,30e") #16a,30e
    bot_plots_ba(edges=n_vertices-1,fontsize=18,bot_positions="bottom",report_fig="16b,30f") #16b,30f

def plot_99_neutral():
    bot_plots(fontsize=18,neutral_bots=True,report_fig="25d") #25d
    bot_plots_ba(edges=n_vertices-1,fontsize=18,bot_positions="top",neutral_bots=True,report_fig="17a") #17a
    bot_plots_ba(edges=n_vertices-1,fontsize=18,bot_positions="bottom",neutral_bots=True,report_fig="17b") #17b

def plot_249():
    bot_plots(fontsize=18,t_lim=t_lim249,report_fig="24a") #24a
    bot_plots(fontsize=18,t_lim=t_lim249,both_sides=True,report_fig="24b") #24b
    bot_plots(edges=n_vertices-1,fontsize=18,t_lim=t_lim249,report_fig="24c") #24c
    bot_plots_ba(edges=n_vertices-1,fontsize=18,t_lim=t_lim249,report_fig="24d") #24d
    bot_plots_ba(edges=n_vertices-1,fontsize=18,t_lim=t_lim249,bot_positions="top",report_fig="24e") #24e
    bot_plots_ba(edges=n_vertices-1,fontsize=18,t_lim=t_lim249,bot_positions="bottom",report_fig="24f") #24f

def plot_249_neutral():
    bot_plots(fontsize=18,t_lim=t_lim249,neutral_bots=True,report_fig="25f") #25f 
    bot_plots_ba(edges=n_vertices-1,fontsize=18,t_lim=t_lim249,neutral_bots=True,report_fig="25e") #25e 
    bot_plots_ba(edges=n_vertices-1,fontsize=18,t_lim=t_lim249,bot_positions="top",neutral_bots=True,report_fig="25a") #25a 
    bot_plots_ba(edges=n_vertices-1,fontsize=18,t_lim=t_lim249,bot_positions="bottom",neutral_bots=True,report_fig="25b") #25b 

def plot_altf():
    bot_plots_altf(fontsize=18,report_fig="26e,f") #26e,f
    bot_plots_altf_ba(edges=n_vertices-1,fontsize=18,report_fig="26c,d") #26c,d
    bot_plots_altf_ba(edges=n_vertices-1,bot_positions="top",fontsize=18,report_fig="18a,b") #18a,b
    bot_plots_altf_ba(edges=n_vertices-1,bot_positions="top",t_lim=t_lim249,fontsize=18,report_fig="26a,b") #26a,b

def plot_dynamic():
    bot_plots_ba_dynamic(edges=n_vertices-1,epsilon=1,fontsize=18,report_fig="19a,31a") #19a,31a
    bot_plots_ba_dynamic(edges=n_vertices-1, epsilon=0.6,fontsize=18,report_fig="27a") #27a
    bot_plots_ba_dynamic(edges=n_vertices-1,epsilon=1,neutral_bots=True,fontsize=18,report_fig="27e") #27e
    bot_plots_ba_dynamic(edges=n_vertices-1, epsilon=0.6,neutral_bots=True,fontsize=18,report_fig="21a,31e") #21a,31e
    bot_plots_ba_dynamic(edges=n_vertices-1,epsilon=1,both_sides=True,fontsize=18,report_fig="31c,28a") #31c,28a
    bot_plots_ba_dynamic(edges=n_vertices-1, epsilon=0.6,both_sides=True,fontsize=18,report_fig="27c") #27c


def plot_dynamic_seeking():
    bot_plots_ba_dynamic(edges=n_vertices-1,epsilon=1,fontsize=18,seeking_bots=True,report_fig="19b,31b") #19b,31b
    bot_plots_ba_dynamic(edges=n_vertices-1, epsilon=0.6,fontsize=18,seeking_bots=True,report_fig="27b") #27b
    bot_plots_ba_dynamic(edges=n_vertices-1,epsilon=1,neutral_bots=True,fontsize=18,seeking_bots=True,report_fig="27f") #27f
    bot_plots_ba_dynamic(edges=n_vertices-1, epsilon=0.6,neutral_bots=True,fontsize=18,seeking_bots=True,report_fig="21b 31f") #21b 31f
    bot_plots_ba_dynamic(edges=n_vertices-1,epsilon=1,both_sides=True,fontsize=18,seeking_bots=True,report_fig="31d,28b") #31d,28b
    bot_plots_ba_dynamic(edges=n_vertices-1, epsilon=0.6,both_sides=True,fontsize=18,seeking_bots=True,report_fig="27d") #27d

def plot_shifted():
    bot_plots_shifted(fontsize=18, initial_opinion_range=[-0.75, 1],report_fig="23a,b") #23a,b
    bot_plots_shifted(fontsize=18,initial_opinion_range=[-0.5,1],report_fig="23c,d") #23c,d
    bot_plots_shifted(fontsize=18, initial_opinion_range=[0, 1],report_fig="23e,f") #23e,f

starttime=datetime.now()
plot_99()
#plot_99_neutral()
#plot_249()
#plot_249_neutral()
#plot_altf()
#plot_dynamic()
#plot_dynamic_seeking()
#plot_shifted()
print(datetime.now()-starttime)