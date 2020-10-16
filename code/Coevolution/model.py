import numpy as np
import random

class coevolution_model_base:
    def __init__(self, n_vertices=100, n_edges=50, n_opinions=2, phi=0.5):
        self.vertices = np.random.randint(n_opinions, size=(n_vertices))
        self.adjacency = np.zeros((n_vertices,n_vertices))
        edges =  [[i,j] for i in range(1, n_vertices) for j in range(i)]
        edges = np.array(random.sample(edges,k=n_edges))
        self.adjacency[edges[:,0],edges[:, 1]] = 1

        self.n_vertices = n_vertices
        self.n_edges = n_edges
        self.n_opinions = n_opinions
        self.phi = phi
        self.t = 0

    def step(self):
        vertex = np.random.randint(self.n_vertices)
        if np.sum((self.adjacency+np.transpose(self.adjacency))[vertex]) > 0:
            if np.random.uniform(0,1)>self.phi:
                self.update_edge(vertex)
            else:
                self.update_opinion(vertex)
        self.t +=1
    def update_opinion(self, vertex):
        neighbours = np.arange(self.n_vertices)[(self.adjacency+np.transpose(self.adjacency))[vertex] > 0]
        self.vertices[vertex] = self.vertices[np.random.choice(neighbours)]
    def update_edge(self,vertex):
        same_opinion = self.vertices == self.vertices[vertex]
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
        for c in self.connected_components():
            opinions = set(self.vertices[np.array(c)])
            return len(opinions)<=1




class coevolution_model_general:
    def __init__(self, n_vertices=100, n_edges=50, n_opinions=2, phi=0.5, d=1, connect = lambda x,y: (x==y).flatten(),
                 update = lambda x,y: y,convergence_criterion= lambda x: len(np.unique(x,axis=0))<=1):
        # Connect is expected to work on numpy arrays and return a boolean numpy array of the same dimension as the inputs.
        if n_opinions == 0:
            print("n_opinions set to 0. Using continuous opinions")
            self.vertices = np.random.uniform(size=(n_vertices, d))
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


    def step(self):
        vertex = np.random.randint(self.n_vertices)
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
        for c in self.connected_components():
            return self.convergence_criterion(self.vertices[np.array(c)])



