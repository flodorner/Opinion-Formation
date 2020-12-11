from exp_WBT_model import *
import numpy as np
import matplotlib.pyplot as plt

''' reproduces figure 3 from schweighöfer et al's 2020 paper about hyperpolarization. 
" Opinion change and balance in a 2D opinion space. Opinion exchange between i and j as a function
of oi , with oj fixed (green arrow, [.5, .5]). Each black arrow represents the resulting change in oi , given an
interaction between i and j: The basis of the arrow represents oi before the interaction and the tip of the arrow
is the position of oi after interaction. The background color in panel a) encodes the balance between i and j,
and in panel"
interesting to understand SGM, Aij and evaluative extremeness f'''



e=0.0
b_j=np.array([0.5,0.5])
alpha = 0.3

x=np.linspace(-1,1)
y=np.linspace(-1,1)
X,Y=np.meshgrid(x,y)

def f(x,y):
    
    return recalculate_emotion(np.array([x,y]), b_j, e=e)


Z = np.zeros((50,50))

for i in range(50):
   for j in range(50):
       pass
       #Z[i,j] = f(X[i,j],Y[i,j])

#plt.pcolor(X, Y, Z, shading='auto')
#plt.colorbar()
plt.arrow(0,0,b_j[0],b_j[1],width=0.01,length_includes_head=True)

grid2=50
x=np.linspace(0.0,0.8,grid2)
y=np.linspace(-0.4,0.4,grid2)
x=np.linspace(-1,1,grid2)
y=np.linspace(-1,1,grid2)
X,Y=np.meshgrid(x,y)

def f(x,y):
    
    Aij=recalculate_emotion(np.array([x,y]), b_j, e=e)
    b_i1=exchange_opinion(np.array([x,y]),b_j,Aij, alpha=alpha)
    #plt.arrow(x,y,b_i1[0]-x,b_i1[1]-y)
    dx=b_i1[0]-x
    dy=b_i1[1]-y
    return dx, dy


Z1 = np.zeros((grid2,grid2))
Z2 =np.zeros((grid2,grid2))
L =np.zeros((grid2,grid2))
for i in range(grid2):
   for j in range(grid2):
       Z1[i,j],Z2 [i,j]= f(X[i,j],Y[i,j])
       L[i,j]=np.sqrt(np.sum(np.power([Z1[i,j],Z2 [i,j]], 2)))

#plt.quiver(X, Y, Z1,Z2)

plt.pcolor(X, Y, L)
plt.colorbar()

plt.show()
