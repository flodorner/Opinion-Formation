import numpy as np
import random
import networkx as nx
from collections import deque
from matplotlib import pyplot as plt

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
            #initialize random graph
            self.n_edges = n_edges
            self.graph = nx.gnm_random_graph(n_vertices, n_edges, seed=None, directed=False)
            '''
            self.adjacency = np.zeros((n_vertices,n_vertices))
            #list of edges, for example [2,5] edge between node 2 and 5. j<i
            edges =  [[i,j] for i in range(1, n_vertices) for j in range(i)]
            edges = np.array(random.sample(edges,k=n_edges))
            #gives lower triangular matrix
            self.adjacency[edges[:,0],edges[:, 1]] = 1
            '''
        elif initial_graph == "barabasi_albert":
            allowed_sizes = np.cumsum(np.arange(n_vertices-1,0,-2))
            index = np.argmin(allowed_sizes<=n_edges) #argmin takes the earliest index if there is a tie
            assert index>0 #Number of edges needs to be at least n_vertices-1
            self.graph = nx.barabasi_albert_graph(n_vertices,index,seed=None)
            self.n_edges = allowed_sizes[index-1]
            if self.n_edges != n_edges:
                print("Amount of edges in BA-graph can only take on some specific values. Using n_edges = " + str(self.n_edges))
        else:
            assert type(initial_graph) != str #Make sure typos in graph generation don't cause problems.
            print("Graph initialized with provided adjacency matrix. n_edges set to " +str (np.sum(initial_graph)))
            self.adjacency = initial_graph
            self.graph = nx.from_numpy_matrix(self.adjacency)
            #self.n_edges = np.sum(initial_graph)


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
        #if np.sum((self.adjacency[vertex]+np.transpose(self.adjacency[:,vertex]))) > 0:
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
        neighbours = np.array(list(self.graph.neighbors(vertex)))
        #neighbours = np.arange(self.n_vertices)[(self.adjacency[vertex] +np.transpose(self.adjacency[:,vertex])) > 0]
        noise = next(self.noise_buffer,None)
        if noise is None:
            self.noise_buffer = (i for i in self.noise_generator((self.n_vertices * self.n_vertices,self.d)))
            noise = next(self.noise_buffer)
        self.vertices[vertex] = self.update(self.vertices[vertex],self.vertices[np.random.choice(neighbours)],noise)
    def update_edge(self,vertex):
        same_opinion = self.connect(self.vertices,self.vertices[vertex])
        same_opinion[vertex] = False
        if np.sum(same_opinion)>0:
            neighbours = np.array(list(self.graph.neighbors(vertex)))
            #neighbours = np.arange(self.n_vertices)[(self.adjacency[vertex]+np.transpose(self.adjacency[:,vertex])) > 0]
            old_neighbour = np.random.choice(neighbours)
            new_neighbour = np.random.choice(np.arange(self.n_vertices)[same_opinion])
            self.graph.add_edge(vertex, new_neighbour)
            if self.graph.number_of_edges() > self.n_edges:
                self.graph.remove_edge(vertex, old_neighbour)
            
            '''
            if new_neighbour>vertex:
                self.adjacency[new_neighbour,vertex] = 1
            else:
                self.adjacency[vertex, new_neighbour] = 1
            if np.sum(self.adjacency)>self.n_edges:
                if old_neighbour > vertex:
                    self.adjacency[old_neighbour, vertex] = 0
                else:
                    self.adjacency[vertex, old_neighbour] = 0
            '''


    def connected_components(self):
        '''
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
        '''
        return list(nx.connected_components(self.graph))

    def convergence(self):
        return self.convergence_criterion(self)

    def draw_graph(self, path):
        """draws output and saves it, needs NetworkX graph self.graph"""
        drawing = self.graph
        pos=nx.spring_layout(drawing)
        plt.figure(figsize=(10,10))
        nx.draw(drawing,pos,node_size=20,alpha=0.5,node_color="blue", with_labels=False)
        #nx.draw_networkx_labels(drawing,pos,font_size=20,font_family='sans-serif')
        #labels = nx.get_edge_attributes(graph,'weight')
        #nx.draw_networkx_edge_labels(graph,pos,edge_labels=labels)
        plt.axis('equal')
        plt.savefig(path)
        plt.close()
        for i in range(self.d):
            res = {idx : self.vertices[idx][i] for idx in range(len(self.vertices))}
            nx.set_node_attributes(self.graph, res, 'opinions'+str(i))
        nx.write_gexf(drawing, path+"Ggexf.gexf")




class holme(coevolution_model_general):
    def __init__(self, n_vertices=100, n_edges=50, n_opinions=0, phi=0.5):
        super().__init__(n_vertices=n_vertices,n_edges=n_edges,n_opinions=n_opinions,phi=phi,d=1
        ,connect=lambda x, y: (x == y).flatten(),update=lambda x, y, noise: y,
        convergence_criterion=lambda x:
        np.all([len(np.unique(x.vertices[np.array(list(c))], axis=0)) <= 1 for c in x.connected_components()])
                         ,systematic_update=False,noise_generator = lambda size: np.zeros(size))
class holme2(coevolution_model_general):
    # using parameters gamma and k of the paper
    # k=2M/N    gamma=10=n_vertices/n_opinions
    def __init__(self, n_opinions=5, phi=0.5, gamma=10, k=4):
        n_vertices=n_opinions*gamma
        n_edges=np.int(n_vertices*k/2)
        super().__init__(n_vertices=n_vertices,n_edges=n_edges,n_opinions=n_opinions,phi=phi,d=1
        ,connect=lambda x, y: (x == y).flatten(),update=lambda x, y, noise: y,
        convergence_criterion=lambda x:
        np.all([len(np.unique(x.vertices[np.array(c)], axis=0)) <= 1 for c in x.connected_components()])
                         ,systematic_update=True,noise_generator = lambda size: np.zeros(size))
        #for method has_changed()
        self.vertices_previous = np.copy(self.vertices)

    def has_changed(self):
        # as an alternative / proxy to computing connected components, 
        # check if any opinions changed since the last time this function was called
        if  np.all(self.vertices_previous == self.vertices): 
            return False
        else: 
            self.vertices_previous = np.copy(self.vertices)
            return True
def sgm(x,y):
    prod = x*y
    return np.sign(prod)*np.sqrt(np.abs(prod))

def update_weighted_balance(x,y,f,alpha,noise):
    attitude = f(np.mean(sgm(x,y)))
    b = sgm(y,attitude)
    return np.clip(x+alpha*(b-x)+noise,-1,1)

class weighted_balance(coevolution_model_general):
    def __init__(self, n_vertices=100, d=3,z=0.01,f=lambda x:x,alpha=0.5):
        super().__init__(n_vertices=n_vertices,n_edges=int(n_vertices*(n_vertices-1)/2),n_opinions=0,phi=0,d=d,
                         update = lambda x,y,noise: update_weighted_balance(x,y,f,alpha,noise),
                         connect = lambda x,y: np.zeros(len(x),dtype=np.bool),
                         convergence_criterion = lambda x: len(x.run_diffs)>=5 and np.all(np.array(x.run_diffs)<z*d*n_vertices)
                         ,systematic_update=True,noise_generator=lambda size:np.random.normal(scale=z,size=size))
        
def connect_weighted_balance(x,y):
    ### if angle between two agents' opinion vectors is less than 90 deg --> connect vertices
    return np.dot(x,y)>0

def connect_weighted_balance_angle(x,y, deg=np.pi/3):
    ### if angle between two agents' opinion vectors is less than deg --> connect vertices
    return np.arccos(np.dot(x,y,axis=1)/(np.linalg.norm(x)*np.linalg.norm(y)))< deg

##d_max = sqrt(4*d)
def connect_weighted_balance_dist(x,y, d=0.5):
    return np.linalg.norm(x-y, axis=1) <= d

        
class weighted_balance_general(coevolution_model_general):
    def __init__(self, n_vertices=100,n_edges=120, d=5,z=0.01,phi=0.6, f=lambda x:np.sign(x)*abs(x)**(1-0.4),alpha=0.4, dist=0):
        super().__init__(n_vertices=n_vertices,n_edges=n_edges,n_opinions=0,phi=phi,d=d,
                         update = lambda x,y,noise: update_weighted_balance(x,y,f,alpha,noise),
                         connect = lambda x,y: connect_weighted_balance_dist(x, y, dist),
                         convergence_criterion = lambda x: len(x.run_diffs)>=5 and np.all(np.array(x.run_diffs)<=z*d*n_vertices*(1-phi)),
                         systematic_update=True,noise_generator=lambda size:np.random.normal(scale=z,size=size))

def update_weighted_balance_bot(x,y,f,alpha,noise):
    if x[-1] == 1:
        return x
    else:
        attitude = f(np.mean(sgm(x[:-1],y[:-1])))
        b = sgm(y[:-1],attitude)
        return np.append(np.clip(x[:-1]+alpha*(b-x[:-1])+noise[:-1],-1,1),x[-1])

class weighted_balance_bots(coevolution_model_general):
    def __init__(self, n_vertices=100, d=3, z=0.01, f=lambda x: x, alpha=0.5,   n_edges=None,initial_graph=None,
                 neutral_bots=False,both_sides=False,bot_positions=None, n_bots=10):
        #n_bots determines the amount of bots deployed (per side)
        #both sides determines whether there are bots for both extreme opinions. This effectively double n_bots.
        #neutral bots determines wheter the bots will have a neutral opinion (constant 0). This cannot be set to True at the same time as both_sides.
        #bot_positions determines which nodes become bots. When it is set to None, the first n_bots nodes become bots,
        # which leads to essentially random placement for randomly generated graphs. If it is "top", the nodes with the highest degree become bots.
        #If it is bottom, the nodes with the lowest degree become bots.
        #When both sides is True, bots are always positioned at the beginning for one and the end for the other side.
        if n_edges is None:
            n_edges = int(n_vertices * (n_vertices - 1) / 2)
        super().__init__(n_vertices=n_vertices,n_edges=n_edges,n_opinions=0,phi=0,d=d+1,
                         update = lambda x,y,noise: update_weighted_balance_bot(x,y,f,alpha,noise),
                         connect = lambda x,y: np.zeros(len(x),dtype=np.bool),
                         convergence_criterion = lambda x: len(x.run_diffs)>=5 and np.all(np.array(x.run_diffs)<z*d*(n_vertices-(n_bots)-n_bots*both_sides))
                         ,systematic_update=True,noise_generator=lambda size:np.random.normal(scale=z,size=size),initial_graph=initial_graph)
        assert not (both_sides and neutral_bots) #we can either have bots on both extremes or neutral bots
        assert not (both_sides and bot_positions!= None) #Positioning bots based on vertex degree only implemented for neutral/one sided bots
        assert bot_positions in [None,"top","bottom"] #bot_positions must be None,"top" or "bottom"

        if bot_positions is None:
            bot_indices = np.array([True for i in range(n_bots)]+[False for i in range(n_vertices-n_bots)])
        elif bot_positions == "top":
            bot_indices = np.zeros(n_vertices,dtype=np.bool)
            bot_nodes = sorted(self.graph.degree(), key=lambda x: x[1])[-n_bots:]
            for node,degree in bot_nodes:
                bot_indices[node] = True
        else:
            bot_indices = np.zeros(n_vertices, dtype=np.bool)
            bot_nodes = sorted(self.graph.degree(), key=lambda x: x[1])[:n_bots]
            for node,degree in bot_nodes:
                bot_indices[node] = True

        self.vertices[bot_indices] = 1
        if neutral_bots and n_bots >0:
            for i in range(n_vertices):
                if bot_indices[i]:
                    self.vertices[i,:-1] = 0
        if both_sides and n_bots>0:
            assert 2*n_bots<n_vertices
            self.vertices[-n_bots:,:-1] = -1
            self.vertices[-n_bots:, -1] = 1

        self.n_bots=n_bots
        self.n_vertices=n_vertices
        self.both_sides=both_sides

def H(O,d):
    s=0
    for i in range(len(O)):
        for j in range(i):
            s += np.linalg.norm(O[i]-O[j],ord=2)**2
    return (1/(4*d))*(4/len(O)**2)*s

