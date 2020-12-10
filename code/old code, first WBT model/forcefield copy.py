from WBT_model import *


e=0.7
b_j=np.array([0.8,0.8,0.8])
alpha = 0.8

x=np.linspace(-1,1)
y=np.linspace(-1,1)
z=np.linspace(-1,1)
X,Y,Z=np.meshgrid(x,y,z)

def f(x,y,z):
    
    return recalculate_emotion(np.array([x,y,z]), b_j, e=e)

grid=50
R = np.zeros((grid,grid,grid))

for i in range(grid):
   for j in range(grid):
       for k in range(grid):
            R[i,j,k] = f(X[i,j,k],Y[i,j,k],Z[i,j,k])
R0=np.logical_and(R<0.1 , R>-0.1)
xs,ys,zs=np.nonzero(R0)

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
plt.title("test")
ax.scatter(X[xs,ys,zs],Y[xs,ys,zs],Z[xs,ys,zs],c="orange")
plt.show()

e=1
e=3
"""
plt.pcolor(X, Y, Z, shading='auto')
plt.colorbar()
plt.arrow(0,0,b_j[0],b_j[1],width=0.01,length_includes_head=True)


x=np.linspace(0.4,0.6,10)
y=np.linspace(0.4,0.6,10)
x=np.linspace(0,1,10)
y=np.linspace(0,1,10)
X,Y=np.meshgrid(x,y)

def f(x,y):
    
    Aij=recalculate_emotion(np.array([x,y]), b_j, e=e)
    b_i1=exchange_opinion(np.array([x,y]),b_j,Aij, alpha=alpha)
    #plt.arrow(x,y,b_i1[0]-x,b_i1[1]-y)
    dx=b_i1[0]-x
    dy=b_i1[1]-y
    return dx, dy


Z1 = np.zeros((10,10))
Z2 =np.zeros((10,10))
for i in range(10):
   for j in range(10):
       Z1[i,j],Z2 [i,j]= f(X[i,j],Y[i,j])

plt.quiver(X, Y, Z1,Z2)




plt.show()
"""
