import numpy as np
from matplotlib import pyplot as plt
from model import weighted_balance_bots,H
import os

dir_path = os.path.dirname(os.path.realpath(__file__))
image_folder = "\\".join(dir_path.split("\\")[:-2]) + "\\doc\\latex\\images\\"

d = np.linspace(0, 1.5,40)
phi = np.linspace(0,1,40)

#X, Y = np.meshgrid(d, phi)

def community(x,y,edges=499,n_bots=0,seeking=False,initial_graph="barabasi_albert"):
    m= weighted_balance_bots(d=3,n_vertices = 500,
                                n_edges=edges, phi=y,alpha=0.4,epsilon=x,initial_graph=initial_graph,n_bots=n_bots,seeking_bots=seeking)
    
    i=0
    while m.convergence() == False and i <100000:
        i +=1
        m.step()

    return H(m.vertices[m.n_bots:,:-1],m.d-1)


def npmap2d(fun, xs, ys, edges,n_bots=0,seeking=False,initial_graph="barabasi_albert"):
 
  Z = np.empty(len(xs) * len(ys))
  i = 0
  for y in ys:
    print(y)
    for x in xs:
      Z[i] = fun(x, y,edges,n_bots=n_bots,seeking=seeking,initial_graph=initial_graph)
      i += 1
  X, Y = np.meshgrid(xs, ys)
  Z.shape = X.shape
  return X, Y, Z

X, Y, Z = npmap2d(community, d, phi, edges=499,initial_graph=None)

fig = plt.figure(figsize=(6, 5))
left, bottom, width, height = 0.1, 0.1, 0.8, 0.8
ax = fig.add_axes([left, bottom, width, height])
cp1 = plt.contourf(X, Y, Z, cmap=plt.cm.get_cmap('plasma'),levels = [0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1])
plt.colorbar(cp1)

ax.set_title('Hyperpolarization H(O)', fontsize=18)

ax.set_xlabel('$\epsilon$', fontsize=18)
ax.set_ylabel('$\phi$', fontsize=18)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)

fig.savefig(image_folder + 'contour_WBT_random.pdf', bbox_inches='tight', pad_inches=0.185, dpi=fig.dpi, transparent=True)



X,Y,Z = npmap2d(community, d, phi, edges=499)

fig = plt.figure(figsize=(6,5))
left, bottom, width, height = 0.1, 0.1, 0.8, 0.8
ax = fig.add_axes([left, bottom, width, height]) 
cp1 = plt.contourf(X, Y, Z, cmap= plt.cm.get_cmap('plasma'),levels = [0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1])
plt.colorbar(cp1)

ax.set_title('Hyperpolarization H(O)', fontsize=18)

ax.set_xlabel('$\epsilon$',fontsize=18)
ax.set_ylabel('$\phi$',fontsize=18)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
    
fig.savefig(image_folder+'contour_WBT_ba.pdf',bbox_inches='tight',pad_inches=0.185, dpi=fig.dpi,transparent=True)


X, Y, Z = npmap2d(community, d, phi, edges=499,n_bots=50)

fig = plt.figure(figsize=(6, 5))
left, bottom, width, height = 0.1, 0.1, 0.8, 0.8
ax = fig.add_axes([left, bottom, width, height])
cp1 = plt.contourf(X, Y, Z, cmap=plt.cm.get_cmap('plasma'),levels = [0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1])
plt.colorbar(cp1)

ax.set_title('Hyperpolarization H(O)', fontsize=18)

ax.set_xlabel('$\epsilon$', fontsize=18)
ax.set_ylabel('$\phi$', fontsize=18)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)

fig.savefig(image_folder+'contour_WBT_50bots_ba.pdf', bbox_inches='tight', pad_inches=0.185, dpi=fig.dpi, transparent=True)


X, Y, Z = npmap2d(community, d, phi, edges=499,n_bots=50,seeking=True)

fig = plt.figure(figsize=(6, 5))
left, bottom, width, height = 0.1, 0.1, 0.8, 0.8
ax = fig.add_axes([left, bottom, width, height])
cp1 = plt.contourf(X, Y, Z, cmap=plt.cm.get_cmap('plasma'),levels = [0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1])
plt.colorbar(cp1)

ax.set_title('Hyperpolarization H(O)', fontsize=18)

ax.set_xlabel('$\epsilon$', fontsize=18)
ax.set_ylabel('$\phi$', fontsize=18)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)

fig.savefig(image_folder+'contour_WBT_50bots_ba_seeking.pdf', bbox_inches='tight', pad_inches=0.185, dpi=fig.dpi, transparent=True)

X, Y, Z = npmap2d(community, d, phi, edges=499,n_bots=200)

fig = plt.figure(figsize=(6, 5))
left, bottom, width, height = 0.1, 0.1, 0.8, 0.8
ax = fig.add_axes([left, bottom, width, height])
cp1 = plt.contourf(X, Y, Z, cmap=plt.cm.get_cmap('plasma'),levels = [0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1])
plt.colorbar(cp1)

ax.set_title('Hyperpolarization H(O)', fontsize=18)

ax.set_xlabel('$\epsilon$', fontsize=18)
ax.set_ylabel('$\phi$', fontsize=18)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)

fig.savefig(image_folder+'contour_WBT_200bots_ba.pdf', bbox_inches='tight', pad_inches=0.185, dpi=fig.dpi, transparent=True)


X, Y, Z = npmap2d(community, d, phi, edges=499,n_bots=200,seeking=True)

fig = plt.figure(figsize=(6, 5))
left, bottom, width, height = 0.1, 0.1, 0.8, 0.8
ax = fig.add_axes([left, bottom, width, height])
cp1 = plt.contourf(X, Y, Z, cmap=plt.cm.get_cmap('plasma'),levels = [0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1])
plt.colorbar(cp1)

ax.set_title('Hyperpolarization H(O)', fontsize=18)

ax.set_xlabel('$\epsilon$', fontsize=18)
ax.set_ylabel('$\phi$', fontsize=18)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)

fig.savefig(image_folder+'contour_WBT_200bots_ba_seeking.pdf', bbox_inches='tight', pad_inches=0.185, dpi=fig.dpi, transparent=True)