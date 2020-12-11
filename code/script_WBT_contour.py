import numpy as np
from matplotlib import pyplot as plt
from code.model import weighted_balance_general

## Figure 10
## generalized WBT model
## two contour plots for community size under variation of \phi and \epsilon
## VERY LONG RUNTIME several hours


d = np.linspace(0, 1.5,40)
phi = np.linspace(0,1,40)

def community(x,y,edges=500):
    m= weighted_balance_general(d=5,n_vertices = 250,
                                n_edges=edges, phi=y,alpha=0.4,dist=x)
    
    i=0
    while m.convergence() == False and i <50000:
        
        for counter in range(1000):
          m.step()
          i +=1
    
    return np.max([len(k) for k in m.connected_components()])


def npmap2d(fun, xs, ys, edges):
 
  Z = np.empty(len(xs) * len(ys))
  i = 0
  for y in ys:
    for x in xs:
      Z[i] = fun(x, y,edges)
      i += 1
  X, Y = np.meshgrid(xs, ys)
  Z.shape = X.shape
  return X, Y, Z



X,Y,Z = npmap2d(community, d, phi, edges=500)



fig = plt.figure(figsize=(6,5))
left, bottom, width, height = 0.1, 0.1, 0.8, 0.8
ax = fig.add_axes([left, bottom, width, height]) 
cp1 = plt.contourf(X, Y, Z, cmap= plt.cm.get_cmap('viridis_r'))
plt.colorbar(cp1)


ax.set_title('Size of largest community', fontsize=18)


#ax.xaxis.set_major_formatter(ticker.FormatStrFormatter('%0.1f'))

ax.set_xlabel('$\epsilon$',fontsize=18)
ax.set_ylabel('$\phi$',fontsize=18)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
#plt.show()
    
fig.savefig('contour_WBT_new.pdf',bbox_inches='tight',pad_inches=0.185, dpi=fig.dpi,transparent=True)


X1,Y1,Z1 = npmap2d(community, d, phi,edges=1000)



fig = plt.figure(figsize=(6,5))
left, bottom, width, height = 0.1, 0.1, 0.8, 0.8
ax = fig.add_axes([left, bottom, width, height]) 
cp = plt.contourf(X1, Y1, Z1, cmap= plt.cm.get_cmap('viridis_r'))
plt.colorbar(cp,)



ax.set_title('Size of largest community', fontsize=18)


#ax.xaxis.set_major_formatter(ticker.FormatStrFormatter('%0.1f'))

ax.set_xlabel('$\epsilon$',fontsize=18)
ax.set_ylabel('$\phi$',fontsize=18)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
#plt.show()
fig.savefig('contour_WBT_D1000_new.pdf',bbox_inches='tight',pad_inches=0.185, dpi=fig.dpi,transparent=True)

