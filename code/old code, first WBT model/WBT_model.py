import numpy as np
from scipy.stats.mstats import gmean
from random import shuffle, choice, choices
import itertools
import matplotlib.pyplot as plt


''' 
Needs some commenting of the code, otherwise its hard to understand
'''


#####################################################################

''' Model-specfic functions: '''

# N is the number of agents
def create_emotion_mat(N):
    A= np.random.rand(N,N) # initiate interpersonal attitude matrix 
    
    np.fill_diagonal(A, 1)
    
    return A

# S is the number of opinion dimensions
def create_opinion_mat(N,S): 
    return np.random.uniform(low=-1,high=1,size=(N,S)) # initiate opinion matrix

# needed for calculating interpersonal attitude and balanced opionion vector
def signed_geo_mean(a,b):
    return(np.sign(a)*np.sign(b)*((abs(a)*abs(b))**(1./2.)))


# calculate interpersonal Attitude using the sigmoidal function
# b_i and b_j are the opinions of the agents
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
    
# update opinion of i (b_i) depending on j (b_j), 
# interpersonal Attitude Aij, and factor alpha
def exchange_opinion(b_i, b_j, Aij, alpha=0.4):
    
    len_opinions=len(b_i)
    
    B = [signed_geo_mean(b_j[k],Aij) for k in range(len_opinions)]
    
    approx = alpha*(B - b_i)
    
    return b_i + approx


def add_Noise(v,sigma=0.01):
    return(np.clip(v + np.random.normal(0,sigma, size=len(v)),-1,1))


# Define Hyperpolarization measure H
def H(O):
    N=O.shape[0]
    D=O.shape[1]
    s=0
    for i in range(N):
        for j in range(i):
            s += np.linalg.norm(O[i]-O[j],ord=2)**2
    return (1/(4*D))*(4/N**2)*s



#########################################################

''' Run models: '''

# S = Number of opinion dimensions
# N = Number of agents
# e = evaluate extremness

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
        
            # choose another random agent
            _ = randomized_agents.copy()
            _.remove(i)
            j = choice(_)
        
            b_i = O[i]
            b_j = O[j]
            
            # calculate interpersonal attitude Aij
            Aij = recalculate_emotion(b_i, b_j, e=e)
        
            # update agents i's opinion
            b_i_updated = exchange_opinion(b_i, b_j, Aij)   
            
            # O[i]= b_i_updated
            O[i] = add_Noise(b_i_updated,sigma=sigma)
            
            # update agent i's perception of j
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
    
    
    
################################################################

''' Plots: '''


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

##### test parameters 

e_list = list()
for e in [0,0.2,0.4,0.8,1]:
    model_= run_model(N=500, S=3, T=100, e=e,sigma=0.01, conv=False)
    e_list.append(model_[3])

x=range(100)
y1=e_list[0][:-1]
y2 = e_list[1][:-1]
y3= e_list[2][:-1]
y4= e_list[3][:-1]
y5= e_list[4][:-1]
plt.plot(x, y1,label='e=0')
plt.plot(x,y2,label='e=0.2')
plt.plot(x,y3,label='e=0.4')
plt.plot(x,y4,label='e=0.8')
plt.plot(x,y5,label='e=1')
plt.xlabel('t',fontsize=18)
plt.ylabel('H(O)',fontsize=18)
plt.legend(prop={'size': 16})
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)

plt.savefig('testing_e.pdf',bbox_inches='tight',pad_inches=0.185,dpi=1000,transparent=True)
plt.close()


N_list = list()
for N in [20,50,100,500,1000]:
    model_= run_model(N=N, S=3, T=100, e=0.4,sigma=0.01, conv=False)
    N_list.append(model_[3])

x=range(100)
y1=N_list[0][:-1]
y2=N_list[1][:-1]
y3=N_list[2][:-1]
y4=N_list[3][:-1]
y5=N_list[4][:-1]
plt.plot(x, y1,label='N=20')
plt.plot(x,y2,label='N=50')
plt.plot(x,y3,label='N=100')
plt.plot(x,y4,label='N=500')
plt.plot(x,y5,label='N=1000')
plt.xlabel('t',fontsize=18)
plt.ylabel('H(O)',fontsize=18)
plt.legend(prop={'size': 16})
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)


plt.savefig('testing_N.pdf',bbox_inches='tight',pad_inches=0.185,dpi=1000,transparent=True)
plt.close()

plt.show()

D_list = list()
for S in [2,3,5,10,12]:
    model_= run_model(N=500, S=S, T=100, e=0.4,sigma=0.01, conv=False)
    D_list.append(model_[3])

x=range(100)
y1=D_list[0][:-1]
y2 = D_list[1][:-1]
y3= D_list[2][:-1]
y4= D_list[3][:-1]
y5= D_list[4][:-1]
plt.plot(x, y1,label='D=2')
plt.plot(x,y2,label='D=3')
plt.plot(x,y3,label='D=5')
plt.plot(x,y4,label='D=10')
plt.plot(x,y5,label='D=12')
plt.xlabel('t',fontsize=18)
plt.ylabel('H(O)',fontsize=18)
plt.legend(prop={'size': 16})
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)


plt.savefig('testing_D.pdf',bbox_inches='tight',pad_inches=0.185,dpi=1000,transparent=True)
plt.close()


z_list = list()
for z in [0,0.001, 0.01,0.05,0.1]:
    model_= run_model(N=500, S=3, T=100, e=0.4,sigma=z, conv=False)
    z_list.append(model_[3])

x=range(100)
y1=z_list[0][:-1]
y2 = z_list[1][:-1]
y3= z_list[2][:-1]
y4= z_list[3][:-1]
y5= z_list[4][:-1]
plt.plot(x, y1,label='z=0')
plt.plot(x,y2,label='z=0.001')
plt.plot(x,y3,label='z=0.01')
plt.plot(x,y4,label='z=0.05')
plt.plot(x,y5,label='z=0.1')
plt.xlabel('t',fontsize=18)
plt.ylabel('H(O)',fontsize=18)
plt.legend(prop={'size': 16})
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)


plt.savefig('testing_z.pdf',bbox_inches='tight',pad_inches=0.185,dpi=1000,transparent=True)
plt.close()
