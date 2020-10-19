import numpy as np
import random



class coevolution_model_general:
    def __init__(self, n_vertices, n_edges, n_opinions, phi, d, connect,update,convergence_criterion,systematic_update):
        # n_vertices controls the graph size, n_edges the amount of edges in the graph.
        # n_opinions specifies the amount of opinions per dimension. If it is set to 0, opinions are continuous
        # phi is the probability of updating an opinion rather than the graph. d is the amount of opinion dimensions.
        # Connect should receive an array with the first dimension representing nodes and the second opinion dimensions
        # and another array representing the opinion dimensions of a selected node. It then returns a boolean
        # array that indicates whether or not the selected node can become connected to respective other nodes.
        # Update should receive two opinion vectors for single nodes and return a new opinion vector that represents the
        # updated opinion for the first node.
        # Convergence criterion is a function of the model class and should return a boolean indicating whether or not
        # the simulation has converged.
        # Systematic update determines whether nodes are updated (True)
        # or whether the updated node is sampled randomly at each step (False)

        #TODO: History. What norm is used?


        if n_opinions == 0:
            print("n_opinions set to 0. Using continuous opinions")
            self.vertices = np.random.uniform(-1,1,size=(n_vertices, d))
        else:
            self.vertices = np.random.randint(n_opinions, size=(n_vertices,d))

        self.adjacency = np.zeros((n_vertices,n_vertices))
        edges =  [[i,j] for i in range(1, n_vertices) for j in range(i)]
        edges = np.array(random.sample(edges,k=n_edges))
        self.adjacency[edges[:,0],edges[:, 1]] = 1

        self.d = d
        self.connect = connect
        self.update = update
        self.convergence_criterion = convergence_criterion
        self.n_vertices = n_vertices
        self.n_edges = n_edges
        self.n_opinions = n_opinions
        self.phi = phi
        self.t=0
        self.systematic_update = systematic_update
        if self.systematic_update:
            self.buffer = (i for i in range(0))

    def step(self):
        if not self.systematic_update:
            vertex = np.random.randint(self.n_vertices)
        else:
            vertex = next(self.buffer,None)
            if vertex == None:
                self.buffer = (i for i in np.random.shuffle(np.arange(self.n_vertices)))
                vertex = next(self.buffer)
        if np.sum((self.adjacency+np.transpose(self.adjacency))[vertex]) > 0:
            if np.random.uniform(0,1)>self.phi:
                self.update_edge(vertex)
            else:
                self.update_opinion(vertex)
        self.t += 1


    def update_opinion(self, vertex):
        neighbours = np.arange(self.n_vertices)[(self.adjacency+np.transpose(self.adjacency))[vertex] > 0]
        self.vertices[vertex] = self.update(self.vertices[vertex],self.vertices[np.random.choice(neighbours)])
    def update_edge(self,vertex):

        same_opinion = self.connect(self.vertices,self.vertices[vertex])
        same_opinion[vertex] = False
        if np.sum(same_opinion)>0:
            neighbours = np.arange(self.n_vertices)[(self.adjacency+np.transpose(self.adjacency))[vertex] > 0]
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
        ,connect=lambda x, y: (x == y).flatten(),update=lambda x, y: y,
        convergence_criterion=lambda x:
        np.all([len(np.unique(x.vertices[np.array(c)], axis=0)) <= 1 for c in x.connected_components()])
                         ,systematic_update=False)

def sgm(x,y):
    return np.sign(x)*np.sign(y)*np.sqrt(np.abs(x),np.abs(y))

def update_weighted_balance(x,y,f,alpha,z):
    attitude = f(np.mean(sgm(x,y)))
    b = sgm(x,attitude)
    return x+alpha*(b-x)+np.random.normal(scale=z,size=x.shape)


class weighted_balance(coevolution_model_general):
    def __init__(self, n_vertices=100, d=1,z=0.01,f=lambda x:x,alpha=0.01):
        super().__init__(n_vertices=n_vertices,n_edges=n_vertices*(n_vertices-1)/2,n_opinions=0,phi=1,d=d,
                         update = lambda x,y: update_weighted_balance(x,y,f,alpha,z),
                         connect = lambda x,y: np.zeros(y.shape),
                         convergence_criterion = lambda x: x.t<1000,systematic_update=True)


