import numpy as np
import random
from collections import deque


class coevolution_model_general:
    def __init__(self, n_vertices, n_edges, n_opinions, phi, d, connect,update,convergence_criterion,systematic_update,noise_generator,initial_graph=None):
        # n_vertices controls the graph size, n_edges the amount of edges in the graph.
        # n_opinions specifies the amount of opinions per dimension. If it is set to 0, opinions are continuous
        # phi is the probability of updating an opinion rather than the graph. d is the amount of opinion dimensions.
        # Connect is a function and should receive an array with the first dimension representing nodes and the second opinion dimensions
        # and another array representing the opinion dimensions of a selected node. It then returns a boolean
        # array that indicates whether or not the selected node can become connected to respective other nodes.
        # Update is a function and should receive two opinion vectors for single nodes as well as a noise term
        # and return a new opinion vector that represents the updated opinion for the first node.
        # Convergence criterion is a function of the model class and should return a boolean indicating whether or not
        # the simulation has converged.
        # Systematic update determines whether nodes are updated (True)
        # or whether the updated node is sampled randomly at each step (False)
        # Noise generator is a function that creates a vector of size n containing independent noise given n.
        # Initial graph is an optional parameter that specifies the initial adjacency matrix
        # (in lower triangular form with empty diagonal) If it is provided, n_edges is overwritten by the amount
        # of edges specified in the matrix.
        if n_opinions == 0:
            #print("n_opinions set to 0. Using continuous opinions")
            self.vertices = np.random.uniform(-1,1,size=(n_vertices, d))
        else:
            self.vertices = np.random.randint(n_opinions, size=(n_vertices,d))

        if initial_graph == None:
            self.n_edges = n_edges
            self.adjacency = np.zeros((n_vertices,n_vertices))
            edges =  [[i,j] for i in range(1, n_vertices) for j in range(i)]
            edges = np.array(random.sample(edges,k=n_edges))
            self.adjacency[edges[:,0],edges[:, 1]] = 1
        else:
            print("Graph initialized with provided adjacency matrix. n_edges set to " +str (np.sum(initial_graph)))
            self.adjacency = initial_graph
            self.n_edges = np.sum(initial_graph)

        self.d = d
        self.connect = connect
        self.update = update
        self.convergence_criterion = convergence_criterion
        self.n_vertices = n_vertices
        self.n_opinions = n_opinions
        self.phi = phi
        self.t=0
        self.systematic_update = systematic_update
        self.index_buffer = (i for i in range(0))
        self.uniform_buffer = (i for i in range(0))
        self.noise_buffer = (i for i in range(0))
        self.vertices_old = np.copy(self.vertices)
        self.run_diffs = deque([],5)
        self.noise_generator = noise_generator


    def step(self):
        if self.t>0 and self.t%self.n_vertices==0:
            self.run_diffs.append(np.sum(np.abs(self.vertices-self.vertices_old)))
            self.vertices_old = np.copy(self.vertices)
        vertex = next(self.index_buffer, None)
        if vertex == None:
            if not self.systematic_update:
                self.index_buffer = (i for i in np.random.randint(0,self.n_vertices,size=self.n_vertices*self.n_vertices))
            else:
                self.index_buffer = (i for i in np.random.permutation(np.arange(self.n_vertices)))
            vertex = next(self.index_buffer)
        if np.sum((self.adjacency[vertex]+np.transpose(self.adjacency[:,vertex]))) > 0:
            if not self.phi==0:
                draw = next(self.uniform_buffer, None)
                if draw == None:
                    self.uniform_buffer = (i for i in np.random.uniform(0,1,size=self.n_vertices*self.n_vertices))
                    draw = next(self.uniform_buffer)
                if draw<self.phi:
                    self.update_edge(vertex)
                else:
                    self.update_opinion(vertex)
            else:
                self.update_opinion(vertex)
        self.t += 1


    def update_opinion(self, vertex):
        neighbours = np.arange(self.n_vertices)[(self.adjacency[vertex] +np.transpose(self.adjacency[:,vertex])) > 0]
        noise = next(self.noise_buffer,None)
        if noise is None:
            self.noise_buffer = (i for i in self.noise_generator((self.n_vertices * self.n_vertices,self.d)))
            noise = next(self.noise_buffer)
        self.vertices[vertex] = self.update(self.vertices[vertex],self.vertices[np.random.choice(neighbours)],noise)
    def update_edge(self,vertex):
        same_opinion = self.connect(self.vertices,self.vertices[vertex])
        same_opinion[vertex] = False
        if np.sum(same_opinion)>0:
            neighbours = np.arange(self.n_vertices)[(self.adjacency[vertex]+np.transpose(self.adjacency[:,vertex])) > 0]
            old_neighbour = np.random.choice(neighbours)
            new_neighbour = np.random.choice(np.arange(self.n_vertices)[same_opinion])
            if new_neighbour>vertex:
                self.adjacency[new_neighbour,vertex] = 1
            else:
                self.adjacency[vertex, new_neighbour] = 1
            if np.sum(self.adjacency)>self.n_edges:
                if old_neighbour > vertex:
                    self.adjacency[old_neighbour, vertex] = 0
                else:
                    self.adjacency[vertex, old_neighbour] = 0

    def connected_components(self):
        A =  np.matrix(self.adjacency+np.transpose(self.adjacency)+np.eye(self.n_vertices))>0
        B = np.zeros((self.n_vertices,self.n_vertices))
        while np.any(A != B):
            B = A.copy()
            A = A*A
        A = np.array(A)
        components = []
        connected_nodes = []
        for i in range(self.n_vertices):
            if len(connected_nodes)==self.n_vertices:
                break
            elif i in connected_nodes:
                continue
            else:
                components.append([j for j in range(self.n_vertices) if A[i][j]>0])
                connected_nodes += components[-1]
        return components
    def convergence(self):
        return self.convergence_criterion(self)



class holme(coevolution_model_general):
    def __init__(self, n_vertices=100, n_edges=50, n_opinions=2, phi=0.5):
        super().__init__(n_vertices=n_vertices,n_edges=n_edges,n_opinions=n_opinions,phi=phi,d=1
        ,connect=lambda x, y: (x == y).flatten(),update=lambda x, y, noise: y,
        convergence_criterion=lambda x:
        np.all([len(np.unique(x.vertices[np.array(c)], axis=0)) <= 1 for c in x.connected_components()])
                         ,systematic_update=False,noise_generator = lambda size: np.zeros(size))

def sgm(x,y):
    prod = x*y
    return np.sign(prod)*np.sqrt(np.abs(prod))

def update_weighted_balance(x,y,f,alpha,noise):
    attitude = f(np.mean(sgm(x,y)))
    b = sgm(y,attitude)
    return np.clip(x+alpha*(b-x)+noise,-1,1)

def evaluative_ex(x,e):
    return np.sign(x)*abs(x)**(1-e)


class weighted_balance(coevolution_model_general):
    def __init__(self, n_vertices=100, d=1,z=0.01,f=lambda x:x,alpha=0.5):
        super().__init__(n_vertices=n_vertices,n_edges=int(n_vertices*(n_vertices-1)/2),n_opinions=0,phi=0,d=d,
                         update = lambda x,y,noise: update_weighted_balance(x,y,f,alpha,noise),
                         connect = lambda x,y: np.zeros(len(x),dtype=np.bool),
                         convergence_criterion = lambda x: len(x.run_diffs)>=5 and np.all(np.array(x.run_diffs)<z*d*n_vertices)
                         ,systematic_update=True,noise_generator=lambda size:np.random.normal(scale=z,size=size))

def update_weighted_balance_bot(x,y,f,alpha,noise):
    if x[-1] == 1:
        return x
    else:
        attitude = f(np.mean(sgm(x[:-1],y[:-1])))
        b = sgm(y[:-1],attitude)
        return np.append(np.clip(x[:-1]+alpha*(b-x[:-1])+noise[:-1],-1,1),x[-1])

class weighted_balance_bots(coevolution_model_general):
    def __init__(self, n_vertices=100, d=1, z=0.01, f=lambda x: x, alpha=0.5, n_bots=10, both_sides=False,
                 neutral_bots=False, n_edges=None):
        if n_edges is None:
            n_edges = int(n_vertices * (n_vertices - 1) / 2)
        super().__init__(n_vertices=n_vertices,n_edges=n_edges,n_opinions=0,phi=0,d=d+1,
                         update = lambda x,y,noise: update_weighted_balance_bot(x,y,f,alpha,noise),
                         connect = lambda x,y: np.zeros(len(x),dtype=np.bool),
                         convergence_criterion = lambda x: len(x.run_diffs)>=5 and np.all(np.array(x.run_diffs)<z*d*(n_vertices-(n_bots)-n_bots*both_sides))
                         ,systematic_update=True,noise_generator=lambda size:np.random.normal(scale=z,size=size))
        self.vertices[:n_bots]=1
        assert not (both_sides and neutral_bots)
        if neutral_bots and n_bots >0:
            self.vertices[:n_bots,:-1] = 0
        if both_sides and n_bots>0:
            assert 2*n_bots<n_vertices
            self.vertices[-n_bots:,:-1] = -1
            self.vertices[-n_bots:, -1] = 1

def H(O,d):
    s=0
    for i in range(len(O)):
        for j in range(i):
            s += np.linalg.norm(O[i]-O[j],ord=2)**2
    return (1/(4*d))*(4/len(O)**2)*s
