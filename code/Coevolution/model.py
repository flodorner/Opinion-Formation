import numpy as np
import random
import networkx as nx
from collections import deque
from matplotlib import pyplot as plt

class coevolution_model_general:
    '''Models the coevolution of opinions within a network of interacting nodes and the shape of the network '''
    def __init__(self, n_vertices, n_edges, n_opinions, d, phi, connect, update,
                 convergence_criterion, systematic_update, noise_generator, initial_graph=None):
        '''Initialization of Graph and Opinions
        ---
        n_vertices (int) : controls the graph size \\
        n_edges (int) : amount of edges in the graph \\
        n_opinions (int) : amount of opinions per dimension. If 0, opinions are continuous \\
        d (int) : amount of opinion dimensions \\
        phi (float, [0,1]) : probability of updating an opinion rather than the graph \\
        connect (fun) : a function with two arrays as input; 1st array has two dimensions: one for nodes and one for the opinions; 2nd array opinions of a selected node. Returns a array that indicates whether or not the selected node can become connected to respective other nodes. \\
        update (fun) : a function and should receive two opinion vectors for single nodes as well as a noise term
        and return a new opinion vector that represents the updated opinion for the first node. \\
        convergence_criterion (fun): a function of the model class and should return a boolean indicating whether or not the simulation has converged. \\
        systematic_update (bool) : determines whether nodes are updated systematically (True) or randomly (False) at each step  \\
        noise_generator (fun) : a function that creates a vector of size n containing independent noise given n as input. \\
        initial_graph (array) (optional) : an initial adjacency matrix (in lower triangular form with empty diagonal). 
        If it is provided, n_edges is overwritten by the amountof edges specified in the matrix. \\
        '''
        if n_opinions == 0:
            #print("n_opinions set to 0. Using continuous opinions")
            self.vertices = np.random.uniform(-1,1,size=(n_vertices, d))
        else:
            self.vertices = np.random.randint(n_opinions, size=(n_vertices,d))

        if initial_graph == None:
            # initialize random graph
            self.n_edges = n_edges
            self.graph = nx.gnm_random_graph(n_vertices, n_edges, seed=None, directed=False)
        elif initial_graph == "barabasi_albert":
            # Creating graph using Barabási-Albert preferential attachment model
            allowed_sizes = np.cumsum(np.arange(n_vertices-1,0,-2))
            index = np.argmin(allowed_sizes<=n_edges) #argmin takes the earliest index if there is a tie
            assert index>0 #Number of edges needs to be at least n_vertices-1
            self.graph = nx.barabasi_albert_graph(n_vertices,index,seed=None)
            self.n_edges = allowed_sizes[index-1]
            if self.n_edges != n_edges:
                print("Amount of edges in BA-graph can only take on some specific values. Using n_edges = " + str(self.n_edges))
        else:
            # Using grph from input
            assert type(initial_graph) != str #Make sure typos in graph generation don't cause problems.
            print("Graph initialized with provided adjacency matrix. n_edges set to " +str (np.sum(initial_graph)))
            self.adjacency = initial_graph
            self.graph = nx.from_numpy_matrix(self.adjacency)
            self.n_edges = self.graph.number_of_edges()

        self.d = d
        self.connect = connect
        self.update = update
        self.convergence_criterion = convergence_criterion
        self.n_vertices = n_vertices
        self.n_opinions = n_opinions
        self.phi = phi
        self.t=0
        self.systematic_update = systematic_update
        self.index_buffer = (i for i in range(0)) # determines order in which vertices are updated
        self.uniform_buffer = (i for i in range(0)) # determines whether opinion or edge will be updated
        self.noise_buffer = (i for i in range(0))
        self.vertices_old = np.copy(self.vertices)
        self.run_diffs = deque([],5)
        self.noise_generator = noise_generator
        
    def step(self):
        '''Handles the update on one vertex, by calling update_opinion or update_edge'''
        if self.t>0 and self.t%self.n_vertices==0:
            #Saving the total changes in opinions
            self.run_diffs.append(np.sum(np.abs(self.vertices-self.vertices_old)))
            self.vertices_old = np.copy(self.vertices)
        vertex = next(self.index_buffer, None)
        if vertex == None:
            #Creation of new buffers
            if not self.systematic_update:
                self.index_buffer = (i for i in np.random.randint(0,self.n_vertices,size=self.n_vertices*self.n_vertices))
            else:
                self.index_buffer = (i for i in np.random.permutation(np.arange(self.n_vertices)))
            vertex = next(self.index_buffer)
        if self.graph.degree(vertex) > 0: 
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
        '''updates opinions of vertex using the update function from object init'''
        neighbours = np.array(list(self.graph.neighbors(vertex)))
        noise = next(self.noise_buffer,None)
        if noise is None:
            self.noise_buffer = (i for i in self.noise_generator((self.n_vertices * self.n_vertices,self.d)))
            noise = next(self.noise_buffer)
        self.vertices[vertex] = self.update(self.vertices[vertex],self.vertices[np.random.choice(neighbours)],noise)
    
    def update_edge(self,vertex):
        '''changes one existing neighbor of vertex to node with same opinion'''
        same_opinion = self.connect(self.vertices,self.vertices[vertex])
        same_opinion[vertex] = False
        if np.sum(same_opinion)>0:
            neighbours = np.array(list(self.graph.neighbors(vertex)))
            old_neighbour = np.random.choice(neighbours)
            new_neighbour = np.random.choice(np.arange(self.n_vertices)[same_opinion])
            self.graph.add_edge(vertex, new_neighbour)
            if self.graph.number_of_edges() > self.n_edges:
                self.graph.remove_edge(vertex, old_neighbour)

    def connected_components(self):
        return list(nx.connected_components(self.graph))

    def convergence(self):
        '''return boolean for whether converged or not'''
        return self.convergence_criterion(self)

    def draw_graph(self, path):
        '''saves network with opinions in path as GEFX file'''
        for i in range(self.d):
            res = {idx : self.vertices[idx][i] for idx in range(len(self.vertices))}
            nx.set_node_attributes(self.graph, res, 'opinions'+str(i))
        nx.write_gexf(self.graph, path+ str(self.t)+".gexf")




class holme(coevolution_model_general):
    '''Coevolution Model as proposed by Holme and Newmann (2006)
    ---
    vertices can either adopt opinion of neighbor or connect to vertices with same opinion
    '''
    def __init__(self, n_vertices=100, n_edges=50, n_opinions=0, phi=0.5):
        '''n_vertices (int) : controls the graph size \\
        n_edges (int) : amount of edges in the graph \\
        n_opinions (int) : amount of opinions per dimension. If 0, opinions are continuous \\
        phi (float, [0,1]) : probability of updating an opinion rather than the graph'''
        super().__init__(n_vertices=n_vertices,n_edges=n_edges,n_opinions=n_opinions,phi=phi,d=1
        ,connect=lambda x, y: (x == y).flatten(),update=lambda x, y, noise: y,
        convergence_criterion=lambda x:
        np.all([len(np.unique(x.vertices[np.array(list(c))], axis=0)) <= 1 for c in x.connected_components()])
                         ,systematic_update=False,noise_generator = lambda size: np.zeros(size))

class holme2(coevolution_model_general):
    '''Model from Holme and Newmann using parameters gamma and k of the paper'''
    def __init__(self, n_opinions=5, phi=0.5, gamma=10, k=4):
        '''
        k (int) : average degree of nodes \\
        gamma (int) : average number of nodes with certain opinion \\
        Relation is: k=2*n_edges/n_vertices and gamma=10=n_vertices/n_opinions'''
        n_vertices=n_opinions*gamma
        n_edges=np.int(n_vertices*k/2)
        super().__init__(n_vertices=n_vertices,n_edges=n_edges,n_opinions=n_opinions,phi=phi,d=1,
                         connect=lambda x, y: (x == y).flatten(), update=lambda x, y, noise: y,
                         convergence_criterion=lambda x:
                         np.all([len(np.unique(x.vertices[np.array(c)], axis=0)) <= 1 for c in x.connected_components()]),
                         systematic_update=True,noise_generator = lambda size: np.zeros(size))
        self.vertices_previous = np.copy(self.vertices) #used for method has_changed()

    def has_changed(self):
        '''as an alternative / proxy to computing connected components, 
        check if any opinions changed since the last time this function was called'''
        if  np.all(self.vertices_previous == self.vertices): 
            return False
        else: 
            self.vertices_previous = np.copy(self.vertices)
            return True

def sgm(x,y):
    '''Signed Geometric Mean
    ---
    used for attitude calulation \\
    input : two opinion arrays'''
    prod = x*y
    return np.sign(prod)*np.sqrt(np.abs(prod))

def update_weighted_balance(x,y,f,alpha,noise):
    '''Calculates new opinions based on attitude of interacting vertices \\
    x and y : opinion arrays \\
    f (fun) : a monotonously increasing function
    alpha (float) : determines speed of opinion change
    '''
    attitude = f(np.mean(sgm(x,y)))
    b = sgm(y,attitude)
    return np.clip(x+alpha*(b-x)+noise,-1,1)

class weighted_balance(coevolution_model_general):
    '''Weighted Balance Model from paper by Schweighofer et al. where network is complete and does not change'''
    def __init__(self, n_vertices=100, d=3,z=0.01,f=lambda x:x,alpha=0.5):
        super().__init__(n_vertices=n_vertices,n_edges=int(n_vertices*(n_vertices-1)/2),n_opinions=0,phi=0,d=d,
                         update = lambda x,y,noise: update_weighted_balance(x,y,f,alpha,noise),
                         connect = lambda x,y: np.zeros(len(x),dtype=np.bool),
                         convergence_criterion = lambda x: len(x.run_diffs)>=5 and np.all(np.array(x.run_diffs)<z*d*n_vertices)
                         ,systematic_update=True,noise_generator=lambda size:np.random.normal(scale=z,size=size))
        
def connect_weighted_balance(x,y):
    '''connects vertices if angle between two agents' opinion vectors is less than 90 deg --> '''
    return np.dot(x,y)>0

def connect_weighted_balance_angle(x,y, deg=np.pi/3):
    '''connects vertices if angle between two agents' opinion vectors is less than deg'''
    return np.arccos(np.dot(x,y)/(np.linalg.norm(x)*np.linalg.norm(y)))< deg

##d_max = sqrt(4*d)
def connect_weighted_balance_dist(x,y, dis=0.5):
    '''connects vertices if distance between two agents' opinion vectors is less than dis'''
    return np.linalg.norm(x-y, axis=1) < dis

        
class weighted_balance_general(coevolution_model_general):
    '''Generalized Weighted Balance model'''
    def __init__(self, n_vertices=100,n_edges=120, d=5,z=0.01,phi=0.6, f=lambda x:np.sign(x)*abs(x)**(1-0.4),alpha=0.4, dist=0):
        ''' dist (float) : maximal l2 distance of opinions for nodes to be able to connect (connected_weighted_balance_dist)'''
        super().__init__(n_vertices=n_vertices,n_edges=n_edges,n_opinions=0,phi=phi,d=d,
                         update = lambda x,y,noise: update_weighted_balance(x,y,f,alpha,noise),
                         connect = lambda x,y: connect_weighted_balance_dist(x, y, dist),
                         convergence_criterion = lambda x: len(x.run_diffs)>=5 and np.all(np.array(x.run_diffs)< z*d*n_vertices),
                         systematic_update=True,noise_generator=lambda size:np.random.normal(scale=z,size=size))

def update_weighted_balance_bot(x,y,f,alpha,noise):
    '''Calculates new opinions based on attitude of interacting vertices, while leaving the bots unchanged\\
    x and y: opinion arrays \\
    f (fun): a monotonously increasing function
    alpha (float) : determines speed of opinion change
    '''
    if x[-1] == 1:
        return x
    else:
        attitude = f(np.mean(sgm(x[:-1],y[:-1])))
        b = sgm(y[:-1],attitude)
        return np.append(np.clip(x[:-1]+alpha*(b-x[:-1])+noise[:-1],-1,1),x[-1])

class weighted_balance_bots(coevolution_model_general):
    '''Weighted Balance Model with Bots, who do not change their own opinion but can interact with others'''
    def __init__(self, n_vertices=100, d=3, z=0.01, f=lambda x: x, alpha=0.5, n_edges=None,initial_graph=None,
                 neutral_bots=False,both_sides=False,bot_positions=None, n_bots=10):
        '''
        n_bots (int) : amount of bots deployed (per side) \\
        both_sides (bool) : determines whether there are bots for both extreme opinions. If true this effectively double n_bots. \\
        neutral_bots (bool) : determines wheter the bots will have a neutral opinion (constant 0). This cannot be set to True at the same time as both_sides. \\
        bot_positions (str): determines which nodes become bots.
            None (default): first n_bots nodes become bots, which leads to random placement for randomly generated graphs
            "top": nodes with the highest degree become bots.
            "bottom": nodes with the lowest degree become bots.
        When both sides is True, bots are always positioned at the beginning for one and the end for the other side.
        '''
        if n_edges is None:
            n_edges = int(n_vertices * (n_vertices - 1) / 2)
        # initializes with one additional opinion dimension which serves as identification of bot
        super().__init__(n_vertices=n_vertices,n_edges=n_edges,n_opinions=0,phi=0,d=d+1,
                         update = lambda x,y,noise: update_weighted_balance_bot(x,y,f,alpha,noise),
                         connect = lambda x,y: np.zeros(len(x),dtype=np.bool),
                         convergence_criterion = lambda x: len(x.run_diffs)>=5 and np.all(np.array(x.run_diffs)<z*d*(n_vertices-(n_bots)-n_bots*both_sides))
                         ,systematic_update=True,noise_generator=lambda size:np.random.normal(scale=z,size=size),initial_graph=initial_graph)
        assert not (both_sides and neutral_bots) #we can either have bots on both extremes or neutral bots
        assert not (both_sides and bot_positions!= None) #Positioning bots based on vertex degree only implemented for neutral/one sided bots
        assert bot_positions in [None,"top","bottom"] #bot_positions must be None,"top" or "bottom"

        if bot_positions is None or n_bots==0:
            if both_sides:
                self.bot_indices = np.array([True for i in range(n_bots)] + [False for i in range(n_vertices - 2*n_bots)]+[True for i in range(n_bots)])
            else:
                self.bot_indices = np.array([True for i in range(n_bots)]+[False for i in range(n_vertices-n_bots)])
        elif bot_positions == "top":
            self.bot_indices = np.zeros(n_vertices,dtype=np.bool)
            bot_nodes = sorted(self.graph.degree(), key=lambda x: x[1])[-n_bots:]
            for node,degree in bot_nodes:
                self.bot_indices[node] = True
        else:
            self.bot_indices = np.zeros(n_vertices, dtype=np.bool)
            bot_nodes = sorted(self.graph.degree(), key=lambda x: x[1])[:n_bots]
            for node,degree in bot_nodes:
                self.bot_indices[node] = True
        #setting the opinions of bots where last identification 'opinion' is 1
        self.vertices[self.bot_indices] = 1
        if neutral_bots and n_bots >0:
            for i in range(n_vertices):
                if self.bot_indices[i]:
                    self.vertices[i,:-1] = 0
        if both_sides and n_bots>0:
            assert 2*n_bots<n_vertices
            self.vertices[-n_bots:,:-1] = -1

        self.n_bots=n_bots
        self.n_vertices=n_vertices
        self.both_sides=both_sides

def H(O,d):
    '''metric to measure hyperpolarization from Schweighofer et al.'''
    s=0
    for i in range(len(O)):
        for j in range(i):
            s += np.linalg.norm(O[i]-O[j],ord=2)**2
    return (1/(4*d))*(4/len(O)**2)*s

