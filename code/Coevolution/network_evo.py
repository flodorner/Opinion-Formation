from model import coevolution_model_general,holme,weighted_balance_general,weighted_balance_bots,H, weighted_balance
from matplotlib import pyplot as plt
import numpy as np
import os
from datetime import datetime

kw={"n_vertices":25, "d":3}

def experiment_loop(kwarg_dict,model_type=None,t_lim=99999, num_bots = 0):
    np.random.seed=0
    
    if model_type == "Holme":
        A = holme(**kwarg_dict)
    elif model_type == "Weighted Balance":
        A = weighted_balance(**kwarg_dict)
    elif model_type == "Weighted Balance General":
        A = weighted_balance_general(**kwarg_dict)
    elif model_type == "Weighted Balance Bots":
        A = weighted_balance_bots(n_vertices=500, d=3, alpha=0.4, n_edges=499,initial_graph="barabasi_albert",bot_positions="top",f=lambda x:min(x,0) ,n_bots = num_bots)
    else:
        print("using general mode")
        A = coevolution_model_general(**kwarg_dict)
    done = False
    while done == False:
        #(A.t%100==0 and A.t <= 1000) or (A.t%1000==0 and A.t<=10000) or
        #if A.t%5000==0:
        #    A.draw_graph(path = image_folder + str(num_bots) +"Bots")
        A.step()
        if A.t%100==0: #Finding connected components is way more complex than the model dynamics. Only check for convergence every 100 steps.
            done = A.convergence()
        if A.t>t_lim:
            print("Model did not converge")
            done = True
    A.draw_graph(path = image_folder + str(num_bots) +"BotsEnd")        


dir_path = os.path.dirname(os.path.realpath(__file__))
image_folder = "\\".join(dir_path.split("\\")[:-2]) + "\\doc\\latex\\images\\"

bot_selection = [0,50,75,125,175,200,250]
for num in bot_selection:
    experiment_loop(kw,model_type="Weighted Balance Bots", num_bots = num)