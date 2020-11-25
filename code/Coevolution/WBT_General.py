import numpy as np
from matplotlib import pyplot as plt
from model import weighted_balance_general

import networkx as nx

dist = np.linspace(0, 1.5,30)
d=dist
phi = np.linspace(0,1,30)

X, Y = np.meshgrid(dist, phi)

def community(x,y,n_edges=500):
    m= weighted_balance_general(d=5,n_vertices = 250,
                                n_edges=n_edges, phi=y,alpha=0.3,dist=x)
    while m.convergence() == False:
        m.step()
    return np.max([len(k) for k in m.connected_components()])

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
    def graph(self):
        for i in range(len(self.results)):
            self.models[i].draw_graph('general graph {}.jpg'.format(i))
res=resultbuffer()
def dynamics_plot():
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
def dynamics_graph():
    m= weighted_balance_general(d=2,n_vertices = 25,
                                n_edges=50, phi=0.45,alpha=0.3,dist=0.3)
    k=1
    ts=0
    while m.convergence() == False:
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

        for j in range(30):
            ts=ts+1
            for i in range(25):
                
                m.step()
        
        k=k+1
        print(k)
        
        
        


dynamics_plot()
def npmap2d(fun, xs, ys, n_edges=500): 
# call fun for each point on grid (xs,ys)
# return Meshgrids YXZ
# fun(x,y,n_edges)
  Z = np.empty(len(xs) * len(ys))
  i = 0
  for y in ys:
    for x in xs:
      Z[i] = fun(x, y,n_edges)
      print("calc x={} y={}".format(x,y))
      i += 1
  X, Y = np.meshgrid(xs, ys)
  Z.shape = X.shape
  return X, Y, Z

"""
# Fixing random state for reproducibility

np.random.seed(19680801)


def randrange(n, vmin, vmax):
    '''
    Helper function to make an array of random numbers having shape (n, )
    with each number distributed Uniform(vmin, vmax).
    '''
    return (vmax - vmin)*np.random.rand(n) + vmin

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

n = 100

# For each set of style and range settings, plot n random points in the box
# defined by x in [23, 32], y in [0, 100], z in [zlow, zhigh].
for m, zlow, zhigh in [('o', -50, -25), ('^', -30, -5)]:
    xs = randrange(n, 23, 32)
    ys = randrange(n, 0, 100)
    zs = randrange(n, zlow, zhigh)
    ax.scatter(xs, ys, zs, marker=m)

ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')

plt.show()
fig.savefig('test3d.png',bbox_inches='tight',pad_inches=0.185, dpi=fig.dpi,transparent=True)


"""





"""
X,Y,Z = npmap2d(community, d, phi)



fig = plt.figure(figsize=(6,5))
left, bottom, width, height = 0.1, 0.1, 0.8, 0.8
ax = fig.add_axes([left, bottom, width, height]) 
cp = plt.contourf(X, Y, Z, cmap= plt.cm.get_cmap('jet_r'))
plt.colorbar(cp)


ax.set_title('Size of largest community', fontsize=18)


ax.xaxis.set_major_formatter(ticker.FormatStrFormatter('%0.1f'))

ax.set_xlabel('$\epsilon$',fontsize=18)
ax.set_ylabel('$\phi$',fontsize=18)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
#plt.show()
    
fig.savefig('contour_WBT.pdf',bbox_inches='tight',pad_inches=0.185, dpi=fig.dpi,transparent=True)


X1,Y1,Z1 = npmap2d(community, d, phi,n_edges=1000)


fig = plt.figure(figsize=(6,5))
left, bottom, width, height = 0.1, 0.1, 0.8, 0.8
ax = fig.add_axes([left, bottom, width, height]) 
cp = plt.contourf(X1, Y1, Z1, cmap= plt.cm.get_cmap('jet_r'))
plt.colorbar(cp)

ax.set_title('Size of largest community', fontsize=18)

ax.xaxis.set_major_formatter(ticker.FormatStrFormatter('%0.1f'))

ax.set_xlabel('$\epsilon$',fontsize=18)
ax.set_ylabel('$\phi$',fontsize=18)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
#plt.show()
fig.savefig('contour_WBT_D1000.pdf',bbox_inches='tight',pad_inches=0.185, dpi=fig.dpi,transparent=True)
"""