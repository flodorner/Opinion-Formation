from model import coevolution_model_general,holme,weighted_balance,weighted_balance_bots,H
from matplotlib import pyplot as plt
import numpy as np
import os

for fname,f in [("id",lambda x: np.sign(x)*np.abs(x)),("root",lambda x: np.sign(x)*np.abs(x)**(0.5)),("square",lambda x: np.sign(x)*np.abs(x)**(2))]:
    for d in [1, 5]:
        for both_sides in [False,True]:
            for neutral_bots in [False, True]:
                for n_vertices in [25,100]:
                    for n_bots in [0,5]:
                        for f_edges in [0.5,1,2,5]:
                            n_edges = int(np.floor(n_vertices*f_edges))
                            if neutral_bots and both_sides:
                                continue
                            submeans=[]
                            subvars=[]
                            subH = []
                            subtimes = []
                            for i in range(10):
                                A = weighted_balance_bots(n_vertices,n_bots=n_bots+n_bots*(not both_sides),d=d,both_sides=both_sides,f=f,neutral_bots=neutral_bots,n_edges=n_edges)
                                for i in range(200000):
                                    A.step()
                                    if A.convergence():
                                        break
                                if n_bots>0 and both_sides==True:
                                    submeans.append(np.mean(A.vertices[n_bots:-n_bots,:-1],axis=0))
                                    subvars.append(np.var(A.vertices[n_bots:-n_bots,:-1],axis=0))
                                    subH.append(H(A.vertices[n_bots:-n_bots,:-1],d=d))
                                elif n_bots==0:
                                    submeans.append(np.mean(A.vertices[:,:-1],axis=0))
                                    subvars.append(np.var(A.vertices[:,:-1],axis=0))
                                    subH.append(H(A.vertices[:,:-1], d=d))
                                else:
                                    submeans.append(np.mean(A.vertices[n_bots+n_bots*(not both_sides):,:-1],axis=0))
                                    subvars.append(np.var(A.vertices[n_bots+n_bots*(not both_sides):,:-1],axis=0))
                                    subH.append(H(A.vertices[n_bots+n_bots*(not both_sides):,:-1], d=d))
                                subtimes.append(A.t)
                            print("d: ", d, fname, " n_vertices: ",n_vertices, " both sides: ", both_sides, " neutral_bots: ",neutral_bots ," n_bots:" , n_bots,
                                  " n_edges: ", f_edges,
                                " means:",np.mean(submeans)," vars:",np.mean(subvars,axis=0)," H: ",np.mean(subH)," t: ", np.mean(subtimes))

# What happens with smaller graphs? What if we allow them to evolve? (Right now, bots just polarize very effectively)

# Adjust conv criterion when phi is nonzero!
# Lecture about data collection, alte reports.
# What if we shift the opinion data?
# Network evolution