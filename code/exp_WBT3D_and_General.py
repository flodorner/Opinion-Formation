import numpy as np
from matplotlib import pyplot as plt
from model import weighted_balance_general, weighted_balance

import networkx as nx

def scatter3d_plot(opinion_mat,save_f=True,loc='Scatter_3D.pdf',title=""):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlim(-1,1)
    ax.set_ylim(-1,1)
    ax.set_zlim(-1,1)
    ax.set_xticks(np.linspace(-1,1,5))
    ax.set_yticks(np.linspace(-1,1,5))
    ax.set_zticks(np.linspace(-1,1,5))
    ax.tick_params(axis='both', which='major', labelsize=12)
    ax.set_xlabel(r'$d_1$',fontsize=14)
    ax.set_ylabel(r'$d_2$',fontsize=14)
    ax.set_zlabel(r'$d_3$',fontsize=14)
    plt.title(title)
    #ax.axes().set_aspect('equal')
    ax.scatter(opinion_mat[:,0], opinion_mat[:,1], opinion_mat[:,2],c='orange')
    
    if save_f == True:
        fig.savefig(loc,bbox_inches='tight',pad_inches=0.185,dpi=1000,transparent=True)
        plt.close()
    else:
        fig.show()
class resultbuffer:
    def __init__(self):
        self.results=[]
        self.models=[]
    def add_op_mat(self, model):
        self.results.append(model.vertices.copy())
        self.models.append(model)
    def plot(self):
        for i in range(len(self.results)):
            scatter3d_plot(self.results[i],save_f=True,loc='general_evo_{}.pdf'.format(i+1),title="t={}".format(i*50))
    def plotWBT(self):
        for i in range(len(self.results)):
            scatter3d_plot(self.results[i],save_f=True,loc='WBT_evo_{}.pdf'.format(i+1),title="t={}".format(i*50))
    def graph(self):
        for i in range(len(self.results)):
            self.models[i].draw_graph('general graph {}.jpg'.format(i))
res=resultbuffer()
def general_evolution():
    m= weighted_balance_general(d=3,n_vertices = 250,
                                n_edges=500, phi=0.45,alpha=0.3,dist=0.3)
    k=1
    res.add_op_mat(m)
    while m.convergence() == False:
        
        
        for j in range(50):
            
            for i in range(250):
                m.step()
        res.add_op_mat(m)
        k=k+1
        print(k)
        
    res.plot()

def WBT_evolution():
    m= weighted_balance(d=3,n_vertices = 250,f=lambda x: np.sign(x)*np.abs(x)**(1-0.4),
                                 alpha=0.3)
    k=1
    res.add_op_mat(m)
    while m.convergence() == False:
        
        #do a plot every n rounds 
        for j in range(30):
            for i in range(n_vertices):
                m.step()
        res.add_op_mat(m)
        k=k+1
        print(k)
        
    res.plotWBT()
def dynamics_graph():
    n_vertices = 25
    m= weighted_balance_general(d=2,n_vertices = n_vertices,
                                n_edges=n_vertices*2, phi=0.52,alpha=0.3,dist=0.4 ) #f=lambda x:x
    k=1
    ts=0
    #while m.convergence() == False:
    while ts<100:
        res.add_op_mat(m)
        
        
         
        fig=plt.figure(figsize=(4,4))
        ax=fig.add_axes([0.15, 0.1, 0.8, 0.8])
        pos={i:m.vertices[i] for i in range(len(m.vertices)) }
        nx.draw(m.graph,pos,ax=ax,node_size=30,alpha=0.5,node_color="blue", with_labels=False)
        limits=plt.axis('on')
        ax.set_xlim([-1,1])
        ax.set_ylim([-1,1])
        plt.title("t={}".format(ts))
        ax.tick_params(left=True, bottom=True, labelleft=True, labelbottom=True)
        
        plt.savefig('general graph {:03d}.jpg'.format(ts))
        plt.close()
        # how many steps/rounds per image
        for j in range(2):
            ts=ts+1
            for i in range(n_vertices):
                
                m.step()
        
        k=k+1
        print(k)





