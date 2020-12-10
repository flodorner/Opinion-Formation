from WBT_model import *


e=0.0
b_j=np.array([0.7,-0.2])
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

plt.pcolor(X, Y, L, shading='auto')
plt.colorbar()

plt.show()
