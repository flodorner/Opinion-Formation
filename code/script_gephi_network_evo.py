'''
Script to save intermediate and converged graphs from exemplary runs of the various models.

All ouputs are saved to image_folder as GEFX files.
Standard size of each model is 25 nodes and edges and otherwise the default values.
Only for Weighted Balance Bots there are spcific changes.

With 'model_input' the model to be examined is set.
Possible values: "Holme", "Weighted Balance", "Weighted Balance General", "Weighted Balance Bot"

With 'bot_selection' there can be different amounts of bots for the WB Bot model specified.

With 'detailed' set to true also the intermediate graphs are saved, otherwise only the converged
Used for Figure  8, 13
'''
model_input = "Weighted Balance General"
bot_selection = [0,50,75,125,175,200,250]
detailed = True

from model import holme,weighted_balance_general,weighted_balance_bots, weighted_balance
from matplotlib import pyplot as plt
import numpy as np

image_folder=""

def experiment_loop(model_type=None,t_lim=99999, num_bots = 0, detailed_evo = False ):
    '''
    detailed_evo (bool) : determines whether intermediate graphs are saved, too, or only the converged one
    '''
    np.random.seed=0
    if model_type == "Holme":
        kw={"n_vertices":25, "n_edges":25}
        A = holme(**kw)
    elif model_type == "Weighted Balance":
        kw={"n_vertices":50, "d":3}
        A = weighted_balance(**kw)
    elif model_type == "Weighted Balance General":
        kw={"n_vertices":25, "n_edges":25, "d":3, "dist":1}
        A = weighted_balance_general(**kw)
    elif model_type == "Weighted Balance Bots":
        A = weighted_balance_bots(n_vertices=500, d=3, alpha=0.4, n_edges=499,initial_graph="barabasi_albert",bot_positions="top",f=lambda x:min(x,0) ,n_bots = num_bots)
    else:
        print("Possible values for model_type are: 'Holme', 'Weighted Balance', 'Weighted Balance General', 'Weighted Balance Bot'")
        return
        #print("using general mode")
        #kw = {"n_vertices":25, "n_edges":25, "n_opinions": 0, "d": 3, "phi": 0.5, }
        #A = coevolution_model_general(**kw)
    done = False
    A.draw_graph(path = image_folder + str(num_bots) + model_type)
    while done == False:
        A.step()
        if A.t%100==0: #Finding connected components is way more complex than the model dynamics. Only check for convergence every 100 steps.
            done = A.convergence()
            if detailed_evo:
                if  A.t <= 1000 or (A.t%1000==0 and A.t<=10000) or A.t%10000==0:
                    A.draw_graph(path = image_folder + str(num_bots) + model_type)
        if A.t>t_lim:
            print("Model did not converge")
            done = True
    A.draw_graph(path = image_folder + str(num_bots) + model_type)        

for num in bot_selection:
    experiment_loop(model_type=model_input, num_bots = num, detailed_evo=detailed)
    if not model_input == "Weighted Balance Bot":
        break
