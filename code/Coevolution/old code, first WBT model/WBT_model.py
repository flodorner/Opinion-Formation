import numpy as np
from scipy.stats.mstats import gmean
from random import shuffle, choice, choices
import itertools

#from sklearn.preprocessing import StandardScaler
#from sklearn.decomposition import PCA

import matplotlib.pyplot as plt



def create_emotion_mat(N):
    A= np.random.rand(N,N)
    
    np.fill_diagonal(A, 1)
    
    return A

def create_opinion_mat(N,S):
    return np.random.uniform(low=-1,high=1,size=(N,S))


def signed_geo_mean(a,b):
    return(np.sign(a)*np.sign(b)*((abs(a)*abs(b))**(1./2.)))

def weighted_euclidean(a, b, w):
    q = a-b
    return np.sqrt((w*q*q).sum())

def euclidian(a,b):
    q = a-b
    return np.sqrt((q*q).sum())

def geo_mean(v1,v2):
    return gmean([np.linalg.norm(v1)*np.linalg.norm(v2)])



def recalculate_emotion(b_i, b_j, sigmoid=True, e=0.4, weights=None):
    
    len_opinions = len(b_i)
    
    if weights == None:
        weights = np.ones(len_opinions)/len_opinions
        
        
    assert len(weights)==len_opinions
    assert round(sum(weights))==1
    
    weighted_sim = sum( weights[k]*
                     signed_geo_mean(b_i[k], b_j[k]) 
       for k in range(len_opinions)) 
        
    
    if sigmoid==True: 
        return np.sign(weighted_sim)*abs(weighted_sim)**(1-e) 
    else:
        return weighted_sim
    
    
def exchange_opinion(b_i, b_j, Aij, alpha=0.4):
    
    len_opinions=len(b_i)
    
    B = [signed_geo_mean(b_j[k],Aij) for k in range(len_opinions)]
    
    approx = alpha*(B - b_i)
    
    return b_i + approx


def select_pairs(size):
    g =itertools.combinations(range(size),2)
    alist = list(g)
    shuffle(alist)
    return alist

def add_Noise(v,sigma=0.01):
    return(np.clip(v + np.random.normal(0,sigma, size=len(v)),-1,1))



def H(O):
    N=O.shape[0]
    D=O.shape[1]
    s=0
    for i in range(N):
        for j in range(i):
            s += np.linalg.norm(O[i]-O[j],ord=2)**2
    return (1/(4*D))*(4/N**2)*s


def E(O):
    N=O.shape[0]
    D=O.shape[1]
    return abs(O).sum()/(N*D)

def C(O):
    # Pearson correlation matrix of transposed input matrix:
    cmat = np.corrcoef(O.T)
    # Set the diagonal to nan:
    np.fill_diagonal(cmat,np.NaN)
    # Take absolute correlation values:
    cmat = abs(cmat)
    # Fisher Z transformation:
    cmat = np.arctanh(cmat)
    # Average:
    cmean = np.nanmean(cmat)
    # Back-transform to pearson correlation:
    return(np.tanh(cmean))




def run_model(N,S,T=1000,e=0.4,sigma=0.01, conv=True):
    A = create_emotion_mat(N=N)
    O = create_opinion_mat(N=N,S=S)
    
    pol_list = [H(O=O)]
    error = list()


    for t in range(T):
    
        agents = list(range(N))
    
        randomized_agents = choices(agents,k=N)
    
        for i in randomized_agents:
        
            O_old = O.copy()
        
            ##choose another random agent
            _ = randomized_agents.copy()
            _.remove(i)
            j = choice(_)
        
            b_i = O[i]
            b_j = O[j]
        
            Aij = recalculate_emotion(b_i, b_j, e=e)
        
            ##update agents i's opinion
            b_i_updated = exchange_opinion(b_i, b_j, Aij)   
            
            #O[i]= b_i_updated
            O[i] = add_Noise(b_i_updated,sigma=sigma)
            
            ##update agent i's perception of j
            A[i,j] = Aij
            
        
        pol_list.append(H(O=O))
        
        error.append(abs(O- O_old).sum())
        
        if (conv==True) and (len(error) > 4) and all(error[i] < N*S*sigma for i in [-1,-2,-3,-4]) :
            break
        
    return(O,A,len(pol_list),pol_list)


def update_model(A, O, e=0.4, sigma=0.01):
    
    N = A.shape[0]
    
    agents = list(range(N))
    
    randomized_agents = choices(agents,k=N)
    
    for i in randomized_agents:
                
        ##choose another random agent
        _ = randomized_agents.copy()
        _.remove(i)
        j = choice(_)
        
        b_i = O[i]
        b_j = O[j]
        
        Aij = recalculate_emotion(b_i, b_j, e=e)
        
        ##update agents i's opinion
        b_i_updated = exchange_opinion(b_i, b_j, Aij)   
        #O[i]= b_i_updated
        O[i] = add_Noise(b_i_updated,sigma=sigma)
        ##update agent i's perception of j
        A[i,j] = Aij
        
    return(A,O)
    
    
    



def scatter3d_plot(opinion_mat,save_f=True,loc='Scatter_3D.pdf'):
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
    ax.scatter(opinion_mat[:,0], opinion_mat[:,1], opinion_mat[:,2],c='orange')
    if save_f == True:
        fig.savefig(loc,bbox_inches='tight',pad_inches=0.185,dpi=1000,transparent=True)
        plt.close()
    else:
        fig.show()
        
O=create_opinion_mat(N=500,S=3)   
A = create_emotion_mat(N=500)     
scatter3d_plot(O,loc='t1.pdf')

for i in range(9):
    A,O=update_model(A,O,e=0.3, sigma=0.01)
    
scatter3d_plot(O,loc='t10.pdf')
    
for i in range(12):
    A,O=update_model(A,O,e=0.3, sigma=0.01)
    
scatter3d_plot(O,loc='t22.pdf')

for i in range(38):
    A,O=update_model(A,O,e=0.3, sigma=0.01)

scatter3d_plot(O,loc='t60.pdf')
